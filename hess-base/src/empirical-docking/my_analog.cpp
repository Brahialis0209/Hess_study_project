#include "global_optimizer.h"
#include "optimization_methods.h"
#include "LBFGS.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using namespace LBFGSpp;

thread_local static std::string _error_message;

//https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D1%80%D0%BE%D1%8F_%D1%87%D0%B0%D1%81%D1%82%D0%B8%D1%86

double random_in_interval_urd(double a, double b){
//    std::random_device rd;   
//    std::mt19937 gen(rd()); 
//    std::uniform_real_distribution<> dis(a, b);
//    return dis(gen);  
  return random_fl_ab( a,  b) ;
}

Eigen::VectorXd generate_Vector_coord_urd(VectorXd& a, VectorXd& b){ //генерирует вектор. a и b - границы распределения
  int size = a.size();
  VectorXd answer = VectorXd::Zero(size);
  for(int i = 0; i < size; i++){
    answer[i] = random_in_interval_urd(a[i], b[i]);
  }
  return answer;
}

Eigen::VectorXd generate_Vector_velocity_urd(VectorXd& a, VectorXd& b){ //в описании алгоритма начальная скорость частицы -(a[i] - b[i]), (a[i] - b[i])
  int size = a.size();
  VectorXd answer = VectorXd::Zero(size);
  for(int i = 0; i < size; i++){
    answer[i] = random_in_interval_urd(-(a[i] - b[i]), (a[i] - b[i]));
  }
  return answer;
}

Eigen::VectorXd generate_Vector_zero_one_urd(int size){ //для шага алгоритма
  VectorXd answer = VectorXd::Zero(size);
  for(int i = 0; i < size; i++){
    answer[i] = random_in_interval_urd(0, 1);
  }
  return answer;
}


Eigen::VectorXd GenerateCoord( Optimizable_molecule& mol ) {
  int solve_size = mol.encoding.size();
  double size_x = mol.size_x; 
  double size_y = mol.size_y;
  double size_z = mol.size_z;
  double conr[3] = {size_x, size_y, size_z}; 
  double start_conr[3] = {size_x, size_y, size_z};
  int start_pos_attempts = 0; 
  int MAX_START_ATTEMPTS = 40; 
  int MAX_START_SECOND_ATTEMPTS = 80; 
  bool (*box_start_checker)(hess::Molecule *lig, double size_x, double size_y, double size_z, simplified_tree& tr, int& ex_count, const vector<int>& encoding_inv, const Eigen::VectorXd & x);
  box_start_checker = check_exceeded_box_limits_start;
  VectorXd x = VectorXd::Zero(solve_size); 
  bool was_moved = false; 
  int ex_count = 0;
  int conr_id = 0;
  while (true) {
    conr_id = 0;
    start_pos_attempts += 1;
    if (start_pos_attempts > MAX_START_ATTEMPTS && !was_moved) {
      start_conr[0] = size_x / 2;
      start_conr[1] = size_y / 2;
      start_conr[2] = size_z / 2;
      box_start_checker = check_exceeded_box_limits; 
      was_moved = true;
    }
    if (start_pos_attempts > MAX_START_SECOND_ATTEMPTS && was_moved) { 
      throw HessException("Failed to place the ligand in the box. Check box dimensions. Maybe he's too small");
    }
    for (int i = 0; i < solve_size - 3; i++) //генерируем случайные координаты начала
      x[i] = random_angle_ils();
    for (int i = solve_size - 3; i < solve_size; i++) {
      x[i] = random_number_ils(start_conr[conr_id]);
      conr_id++;
    }
    if (!box_start_checker(mol.ligand, size_x, size_y, size_z, mol.tr, ex_count, mol.encoding_inv, x))
      break;
  }
  return x;
}




Eigen::VectorXd Swarm(Optimizable_molecule& mol, int depth, double dif) {
  double w = 1; //константы для алгоритма
  double w_diff = 1e-2;
  double w_limit = 0.4;
  const int warms_count = 200;
  std::array<VectorXd, warms_count> current_cords {}; //8 - количество частиц в рое
  std::array<VectorXd, warms_count> velocity {};
  std::array<VectorXd, warms_count> current_best {};
  std::array<double, warms_count> current_best_value {};
  


  int solve_size = mol.encoding.size(); 
  double size_x = mol.size_x; 
  double size_y = mol.size_y;
  double size_z = mol.size_z;
  double conr[3] = {size_x, size_y, size_z}; 
 
  
  LBFGSParam<double> param;
  LBFGSSolver<double> solver(param);
  param.epsilon = 0.0001;
  param.epsilon_rel = 0.0001;
  double alpha = 0.99995, beta = 0.99995;
  VectorXd bestans = VectorXd::Zero(solve_size); //вектор с лучшим ответом
  double best = 1000.0;
  
//  double start_conr[3] = {size_x, size_y, size_z}; 
//  int conr_id = 0;
//  VectorXd x = VectorXd::Zero(solve_size);
//  int start_pos_attempts = 0; 
//  int MAX_START_ATTEMPTS = 40; 
//  int MAX_START_SECOND_ATTEMPTS = 80; 
//  bool (*box_start_checker)(hess::Molecule *lig, double size_x, double size_y, double size_z, simplified_tree& tr, int& ex_count, const vector<int>& encoding_inv, const Eigen::VectorXd & x);
//  box_start_checker = check_exceeded_box_limits_start;
//  bool was_moved = false; 
//  int ex_count = 0;
//  while (true) {
//    conr_id = 0;
//    start_pos_attempts += 1;
//    if (start_pos_attempts > MAX_START_ATTEMPTS && !was_moved) {
//      start_conr[0] = size_x / 2; //изменяем начальное положение
//      start_conr[1] = size_y / 2;
//      start_conr[2] = size_z / 2;
//      box_start_checker = check_exceeded_box_limits; //проверка - box_start_checker и check_exceeded_box_limits - функция - переменная?
//      was_moved = true;
//    }
//    if (start_pos_attempts > MAX_START_SECOND_ATTEMPTS && was_moved) {
//      throw HessException("Failed to place the ligand in the box. Check box dimensions. Maybe he's too small");
//    }
//    for (int i = 0; i < solve_size - 3; i++) //генерируем случайные координаты начала
//      x[i] = random_angle_ils();
//    for (int i = solve_size - 3; i < solve_size; i++) {
//      x[i] = random_number_ils(start_conr[conr_id]);
//      conr_id++;
//    }
//    if (!box_start_checker(mol.ligand, size_x, size_y, size_z, mol.tr, ex_count, mol.encoding_inv, x))
//      break;
//  }
//  VectorXd a = VectorXd::Zero(solve_size); //интервалы для того, чтобы генерировать векторы в этих пределах
//  VectorXd b = VectorXd::Zero(solve_size); //причём генерируются только начальные 
//  for(int i = 0 ; i < solve_size - 3 ; i++){
////    b[i] = random_in_interval_urd(0, 2*M_PI);
//    b[i] = random_angle_ils();
//  }
//  for(int i = solve_size - 3; i < solve_size; i++){
//    double fxi = 0.2*fabs(x[i]);
////    b[i] = random_in_interval_urd(x[i] - fxi, x[i] + fxi); //известные границы - здесь я так и не разобралась, какие границы ставить
//    b[i] = random_in_interval_urd(x[i] - fxi, x[i] + fxi); 
//    
//    //вариант 1 - x[i] - 0.2*x[i], x[i] + 0.2*x[i]
//    //вариант 2 -  -2*abs(x[i]),  2*abs(x[i]) - потому что...
//    //Но и так и так работает одинаково (выдаёт один и тот же результат)
//    //В пределах a , b будут задаваться координаты начальных (и только начальных) положений частиц роя
//    //Поэтому, при достаточно большом количестве итераций частицы так или иначе соберуться у оптимального положения
//  }
  


  VectorXd r_p = VectorXd::Zero(solve_size); 
  VectorXd r_g = VectorXd::Zero(solve_size); 
  double fx = 1000;
  
//  cout << "step 1\n";

//  for(int j = 0; j < current_best.size(); j++){
//
//    current_best[j] = generate_Vector_coord_urd(a, b);
//    velocity[j] = generate_Vector_velocity_urd(a, b);
//    current_cords[j] = current_best[j];
//    solver.minimize(mol, current_cords[j], fx);
//    current_best_value[j] = fx;
//  }
  
  
  for(int j = 0; j < current_best.size(); j++){
    velocity[j] = VectorXd::Zero(solve_size); 
    current_cords[j] = VectorXd::Zero(solve_size); 
    current_best[j] = GenerateCoord(mol); 
    for (int k = 0; k < solve_size; k++) {
      current_cords[j][k] = current_best[j][k];
    }
    solver.minimize(mol, current_cords[j], fx);
    current_best_value[j] = fx;
  }
  
  
//  cout << "step 2\n";
  
  
  bool flag = false;
  bestans = current_best[0];
  best = current_best_value[0];
  double eps = 1e-7;
  double phi_p = 1; 
  double phi_g = 1; 
  double k = 0.5;
  double f = 1;
  VectorXd x_old = VectorXd::Zero(solve_size); 
  for(int iteration = 0; iteration < depth; iteration++){
    for(int i = 0; i < current_cords.size(); i++){ //проходим по всем частицам
      r_p = generate_Vector_zero_one_urd(solve_size);
      r_g = generate_Vector_zero_one_urd(solve_size);
      for(int j=0; j < solve_size; j++){
        velocity[i][j] = w*velocity[i][j] + f*r_p[j]*(current_best[i][j] - current_cords[i][j]) + f*r_g[j]*(bestans[j] - current_cords[i][j]);
        current_cords[i][j] = current_cords[i][j]+velocity[i][j];
      }
      
      x_old.noalias() = current_cords[i];
      random_change(current_cords[i], x_old, dif, conr);
      
      solver.minimize(mol, current_cords[i], fx);
      if(fx < current_best_value[i]){
        current_best_value[i] = fx;
        current_best[i] = current_cords[i];
      }
      if(fx < best){
        if(fabs(best - fx) < eps){
          best = fx;
          bestans = current_cords[i];
          flag = true;
          break;
        }
        best = fx;
        bestans = current_cords[i];
      }

    }
    param.epsilon *= alpha;
    param.epsilon_rel *= alpha;
    dif *= beta;
    if(flag) break;

  }
//  cout << "step 3\n";
  return bestans;

}


void swarm_mutate(VectorXd& candidate, VectorXd& candidate_1, double& candidate_e, double& candidate_1_e, Optimizable_molecule& mol,
        std::vector<VectorXd>& current_cords, std::vector<VectorXd>& current_velocity, 
        std::vector<VectorXd>& current_best, std::vector<double>& current_best_value, int step,
        VectorXd& bestans, double& bestf, LBFGSSolver<double>& solver, double* PersonalBest) {
  int solve_size = mol.encoding.size(); 
  int rots_count = mol.rot_bonds_count;
  int entities = 2 + mol.rot_bonds_count;
  if (entities == 0) return;
  int which_int = random_int_ab(0, int(entities - 1));
  size_t which = size_t(which_int);
  srand((unsigned)time(NULL));
  double r_cm;
  r_cm = (double) rand() / (RAND_MAX + 1.0);
  double w_cm;
  w_cm = (double) rand() / (RAND_MAX + 1.0);
  double c1 = 0.99, c2 = 0.99;
  VectorXd tmp_2 = VectorXd::Zero(solve_size);;
  tmp_2.noalias() = candidate;
  double tmp_2_e = candidate_e;
  if (which == 0) {
    for(int i = 0; i < current_cords.size(); i++) {
      //rough local search
      candidate = current_cords[i];
      solver.minimize(mol, candidate, candidate_e);
      //set the personal best(energy value and position);
      if(candidate_e < current_best_value[i] || step <= 18) {
        if (candidate_e < current_best_value[i]) {
          current_best_value[i] = candidate_e;
          current_best[i] = candidate;
        }
        tmp_2.noalias() = candidate;
        tmp_2_e = candidate_e;
        solver.minimize(mol, tmp_2, tmp_2_e);
        
        if (tmp_2_e < PersonalBest[i]) {
          PersonalBest[i] = tmp_2_e;
          current_best_value[i] = tmp_2_e;
          current_best[i] = tmp_2;
        }
        
        //set the global best(energy value and position);
        if(tmp_2_e < bestf){
          bestf = tmp_2_e;
          bestans = tmp_2;
        }
      }
      //update chaotic map
      r_cm = 1.07*(7.86*r_cm-23.31*r_cm*r_cm+28.75*r_cm*r_cm*r_cm-13.302875*r_cm*r_cm*r_cm*r_cm);
      w_cm = 1.07*(7.86*w_cm-23.31*w_cm*w_cm+28.75*w_cm*w_cm*w_cm-13.302875*w_cm*w_cm*w_cm*w_cm);
      
      for(int i = 0; i < current_cords.size(); i++) {
        for (int v = 3; v>0; v--) {
          current_velocity[i][solve_size - v] = w_cm*current_velocity[i][solve_size - v] + 
                  c1*r_cm*(current_best[i][solve_size - v] - current_cords[i][solve_size - v]) + 
                  c2*(1-r_cm)*(bestans[solve_size - v] - current_cords[i][solve_size - v]);
          current_cords[i][solve_size - v] = current_cords[i][solve_size - v]+current_velocity[i][solve_size - v];
        }
      }
    }
    candidate_1_e = bestf;
    candidate_1 = bestans;
    return;
  }
    
  --which;
  //Take part orientation
  if (which == 0) {
      for(int i = 0; i < current_cords.size(); i++) {
        double g_rad = gyration_radius(current_cords[i], mol);
        if (g_rad > epsilon_fl)  {
          //rough local search
          candidate = current_cords[i];
          solver.minimize(mol, candidate, candidate_e);
          //set the personal best(energy value and position);
          if(candidate_e < current_best_value[i] || step <= 18) {
            if (candidate_e < current_best_value[i]) {
              current_best_value[i] = candidate_e;
              current_best[i] = candidate;
            }
            tmp_2.noalias() = candidate;
            tmp_2_e = candidate_e;
            solver.minimize(mol, tmp_2, tmp_2_e);

            if (tmp_2_e < PersonalBest[i]) {
              PersonalBest[i] = tmp_2_e;
              current_best_value[i] = tmp_2_e;
              current_best[i] = tmp_2;
            }

            //set the global best(energy value and position);
            if(tmp_2_e < bestf){
              bestf = tmp_2_e;
              bestans = tmp_2;
            }
          }
          //update chaotic map
          r_cm = 1.07*(7.86*r_cm-23.31*r_cm*r_cm+28.75*r_cm*r_cm*r_cm-13.302875*r_cm*r_cm*r_cm*r_cm);
          w_cm = 1.07*(7.86*w_cm-23.31*w_cm*w_cm+28.75*w_cm*w_cm*w_cm-13.302875*w_cm*w_cm*w_cm*w_cm);

          for(int i = 0; i < current_cords.size(); i++) {
            for (int v = 6; v>3; v--) {
              current_velocity[i][solve_size - v] = w_cm*current_velocity[i][solve_size - v] + 
                      c1*r_cm*(current_best[i][solve_size - v] - current_cords[i][solve_size - v]) + 
                      c2*(1-r_cm)*(bestans[solve_size - v] - current_cords[i][solve_size - v]);
              current_cords[i][solve_size - v] = current_cords[i][solve_size - v]+current_velocity[i][solve_size - v];
            }
          }
        }
      }
      candidate_1_e = bestf;
      candidate_1 = bestans;
      return;
  }
  /*Torsions*/
  --which;
  if (which < rots_count) {
    for(int i = 0; i < current_cords.size(); i++) {
      //rough local search
      candidate = current_cords[i];
      solver.minimize(mol, candidate, candidate_e);
      //set the personal best(energy value and position);
      if(candidate_e < current_best_value[i] || step <= 18) {
        if (candidate_e < current_best_value[i]) {
          current_best_value[i] = candidate_e;
          current_best[i] = candidate;
        }
        tmp_2.noalias() = candidate;
        tmp_2_e = candidate_e;
        solver.minimize(mol, tmp_2, tmp_2_e);
        
        if (tmp_2_e < PersonalBest[i]) {
          PersonalBest[i] = tmp_2_e;
          current_best_value[i] = tmp_2_e;
          current_best[i] = tmp_2;
        }
        
        //set the global best(energy value and position);
        if(tmp_2_e < bestf){
          bestf = tmp_2_e;
          bestans = tmp_2;
        }
      }
      //update chaotic map
      r_cm = 1.07*(7.86*r_cm-23.31*r_cm*r_cm+28.75*r_cm*r_cm*r_cm-13.302875*r_cm*r_cm*r_cm*r_cm);
      w_cm = 1.07*(7.86*w_cm-23.31*w_cm*w_cm+28.75*w_cm*w_cm*w_cm-13.302875*w_cm*w_cm*w_cm*w_cm);
      
      for(int i = 0; i < current_cords.size(); i++) {
        current_velocity[i][which] = w_cm*current_velocity[i][which] + 
                  c1*r_cm*(current_best[i][which] - current_cords[i][which]) + 
                  c2*(1-r_cm)*(bestans[which] - current_cords[i][which]);
        current_cords[i][which] = current_cords[i][which]+current_velocity[i][which];
      }
    }
  }
  candidate_1_e = bestf;
  candidate_1 = bestans;
  return; 
}


Eigen::VectorXd SwarmPSO(Optimizable_molecule& mol, int depth, vector<mc_out>& outs) {
  const int swarm_count = mol.ligand->get_atoms_count()*3;
  int solve_size = mol.encoding.size(); 
  vector<VectorXd> current_cords;
  std::vector<VectorXd> current_velocity;
  std::vector<VectorXd> current_best;
  std::vector<double> current_best_value;
  current_cords.resize(swarm_count, VectorXd::Zero(solve_size));
  current_velocity.resize(swarm_count, VectorXd::Zero(solve_size));
  current_best.resize(swarm_count, VectorXd::Zero(solve_size));
  current_best_value.resize(swarm_count, max_fl);

  double size_x = mol.size_x; 
  double size_y = mol.size_y;
  double size_z = mol.size_z;
  double conr[3] = {size_x, size_y, size_z}; 
 
  LBFGSParam<double> param;
  LBFGSSolver<double> solver(param);
  param.epsilon = 0.0001;
  param.epsilon_rel = 0.0001;
  param.max_iterations = mol.max_bfgs_iterations;
  double alpha = 0.99995, beta = 0.99995;
  VectorXd bestans = VectorXd::Zero(solve_size);
  VectorXd bestans_result = VectorXd::Zero(solve_size);
  double fx = max_fl;
  
  for(int j = 0; j < current_best.size(); j++){
    current_velocity[j] = VectorXd::Zero(solve_size); 
    current_cords[j] = VectorXd::Zero(solve_size); 
    current_cords[j] = GenerateCoord(mol); 
//    solver.minimize(mol, current_cords[j], fx);
    current_best[j].noalias() = current_cords[j];
    current_best_value[j] = fx;
  }
 
  
  bool flag = false;
  double bestf = max_fl;
  double best_e = max_fl;
//  bestans = current_best[0];
//  bestf = current_best_value[0];
  double tmp_e = bestf;
  double eps = 1e-5;
  double k = 0.5;
  double f = 1;
  VectorXd tmp_rough = VectorXd::Zero(solve_size);
  VectorXd tmp = GenerateCoord(mol); 
  tmp_rough.noalias() = tmp; 
  double candidate_1_e = max_fl;
  double candidate_e = max_fl;
  double* PersonalBest = new double[swarm_count];
  double energy = max_fl;
  int count = 0;
  for(int cou = 0; cou < swarm_count; cou++)
    PersonalBest[cou] = 0;
  int num_steps = mol.num_steps;
//  num_steps = 50;
  VectorXd candidate = VectorXd::Zero(solve_size);
  VectorXd candidate_1 = VectorXd::Zero(solve_size);
  for(int iteration = 0; iteration < num_steps; iteration++) {
    mol.assign_hunt_cap();
    candidate.noalias() = tmp_rough;
    candidate_1.noalias() = tmp;
    swarm_mutate(candidate, candidate_1, candidate_e, candidate_1_e, mol, current_cords, current_velocity, current_best, current_best_value, 
            iteration, bestans, bestf, solver, PersonalBest);
    tmp_rough.noalias() = candidate;
    if (iteration == 0 || metropolis_accept(tmp_e, candidate_1_e, mol.temperature)) {
      tmp.noalias() = candidate_1;
      tmp_e = candidate_1_e;
      if (tmp_e < best_e || outs.size() < mol.num_saved_mins) {
        mol.assign_aut_cap();
        solver.minimize(mol, tmp, tmp_e);
        add_to_output_container(outs, tmp, mol, tmp_e); // 20 - max size
        if (tmp_e < best_e) {
          best_e = tmp_e;
          bestans_result.noalias() = tmp;
        }
      }
    }
    
    //only for very promising ones
    if(std::abs(bestf - energy) < 0.0001) {
      count += 1;
      if(count > 350) {
        iteration = num_steps; //break the loop
        count = 0;
      }
    }
    else{
      energy = bestf;
      count = 0;
    }
//    param.epsilon *= alpha;
//    param.epsilon_rel *= alpha;
//    dif *= beta;
  }
  return bestans_result;

}
