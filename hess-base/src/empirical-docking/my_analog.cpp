#include "global_optimizer.h"
#include "optimization_methods.h"
#include "LBFGS.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXf;
using namespace LBFGSpp;

//https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D1%80%D0%BE%D1%8F_%D1%87%D0%B0%D1%81%D1%82%D0%B8%D1%86

double random_in_interval_urd(double a, double b){
    std::random_device rd;   
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(a, b);
    return dis(gen);  
}

Eigen::VectorXd generate_Vector_coord_urd(VectorXd a, VectorXd b){ //генерирует вектор. a и b - границы распределения
  if(a.size() != b.size()) throw "My bad, you tried to generate a random vector, but limits have different sizes (coord)";
  int size = a.size();
  VectorXd answer = VectorXd::Zero(size);
  for(int i = 0; i < size; i++){
    answer[i] = random_in_interval_urd(a[i], b[i]);
  }
  return answer;
}

Eigen::VectorXd generate_Vector_velocity_urd(VectorXd a, VectorXd b){ //в описании алгоритма начальная скорость частицы -(a[i] - b[i]), (a[i] - b[i])
  if(a.size() != b.size()) throw "My bad, you tried to generate a random vector, but limits have different sizes (velocity)";
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




Eigen::VectorXd Swarm(Optimizable_molecule& mol, int depth, double dif) {
  double w = 0.36; //константы для алгоритма
  double f = 1; 

  std::array<VectorXd, 8> current_cords {}; //8 - количество частиц в рое
  std::array<VectorXd, 8> velocity {};
  std::array<VectorXd, 8> current_best {};
  std::array<double, 8> current_best_value {};


  int solve_size = mol.encoding.size(); //размер молекулы (количество переменных - координаты центра, углы поворота 3*N угла поворота связей)
  double size_x = mol.size_x; //размеры box'a 
  double size_y = mol.size_y;
  double size_z = mol.size_z;

  double conr[3] = {size_x, size_y, size_z}; 
  double start_conr[3] = {size_x, size_y, size_z}; //сохраняем начальные значения отдельно

  int conr_id = 0;
  double fx = 1000;

  VectorXd x = VectorXd::Zero(solve_size); //изменяемый вектор 
  
  //насколько я поняла, в этой части алгоритма идёт определение начального положения белка так, что бы он не выходил за границы
  int start_pos_attempts = 0; 
  int MAX_START_ATTEMPTS = 40; 
  int MAX_START_SECOND_ATTEMPTS = 80; 
  bool (*box_start_checker)(hess::Molecule *lig, double size_x, double size_y, double size_z, simplified_tree& tr, int& ex_count, const vector<int>& encoding_inv, const Eigen::VectorXd & x);
  box_start_checker = check_exceeded_box_limits_start;
  bool was_moved = false; 
  int ex_count = 0;

  while (true) {
    conr_id = 0;
    start_pos_attempts += 1;
    if (start_pos_attempts > MAX_START_ATTEMPTS && !was_moved) {
      start_conr[0] = size_x / 2; //изменяем начальное положение
      start_conr[1] = size_y / 2;
      start_conr[2] = size_z / 2;
      box_start_checker = check_exceeded_box_limits; //проверка - box_start_checker и check_exceeded_box_limits - функция - переменная?
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

  LBFGSParam<double> param;
  LBFGSSolver<double> solver(param);
  param.epsilon = 0.0001;
  param.epsilon_rel = 0.0001;
  double alpha = 0.99995, beta = 0.99995;


  VectorXd bestans = VectorXd::Zero(solve_size); //вектор с лучшим ответом
  double best = 1000.0;

  VectorXd a = VectorXd::Zero(solve_size); //интервалы для того, чтобы генерировать векторы в этих пределах
  VectorXd b = VectorXd::Zero(solve_size); //причём генерируются только начальные 
  for(int i = 0 ; i < solve_size - 3 ; i++){
    b[i] = random_in_interval_urd(0, 2*M_PI);
  }
  for(int i = solve_size - 3; i < solve_size; i++){
    b[i] = random_in_interval_urd(x[i] - 0.2*x[i], x[i] + 0.2*x[i]); //известные границы - здесь я так и не разобралась, какие границы ставить
    //вариант 1 - x[i] - 0.2*x[i], x[i] + 0.2*x[i]
    //вариант 2 -  -2*abs(x[i]),  2*abs(x[i]) - потому что...
    //Но и так и так работает одинаково (выдаёт один и тот же результат)
    //В пределах a , b будут задаваться координаты начальных (и только начальных) положений частиц роя
    //Поэтому, при достаточно большом количестве итераций частицы так или иначе соберуться у оптимального положения
  }


  VectorXd r_p = VectorXd::Zero(solve_size); 
  VectorXd r_g = VectorXd::Zero(solve_size); 

  //инициализация
  for(int j = 0; j < current_best.size(); j++){

    current_best[j] = generate_Vector_coord_urd(a, b);
    velocity[j] = generate_Vector_velocity_urd(a, b);
    current_cords[j] = current_best[j];
    solver.minimize(mol, current_cords[j], fx);
    current_best_value[j] = fx;

  }

  bool flag = false; //флаг того, что изменяется меньше, чем на 10^-4
  bestans = current_best[0];
  best = current_best_value[0];
  //сама оптимизация
  for(int iteration = 0; iteration < depth; iteration++){
    for(int i = 0; i < current_cords.size(); i++){ //проходим по всем частицам
      r_p = generate_Vector_zero_one_urd(solve_size);
      r_g = generate_Vector_zero_one_urd(solve_size);
      for(int j; j < solve_size; j++){
        velocity[i][j] = w*velocity[i][j] + f*r_p[j]*(current_best[i][j] - current_cords[i][j]) + f*r_g[j]*(bestans[j] - current_cords[i][j]);
        current_cords[i][j] = current_cords[i][j]+velocity[i][j];
      }
      solver.minimize(mol, current_cords[i], fx);
      if(fx < current_best_value[i]){
        current_best_value[i] = fx;
        current_best[i] = current_cords[i];
      }
      if(fx < best){
        if(best - fx < 0.0001){
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
    if(flag) break;

  }

  return bestans;

}
