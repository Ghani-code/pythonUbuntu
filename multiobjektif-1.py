import random as rn
import numpy as np
import matplotlib.pyplot as plt
import math
import mpi4py
from mpi4py import MPI

# Minimalisasi

# Inisialisasi populasi acak orang tua kromosom/solusi P
def random_population(n_var, n_sol, lb, ub):
    # n_var = nomor variabel
    # n_sol = nomor solusi acak
    # lb = batas bawah
    # ub = batas atas
    pop = np.zeros((n_sol, n_var))
    for i in range(n_sol):
        pop[i,:] = np.random.uniform(lb, ub)

    return pop

# Pada setiap iterasi, dari 2 orang tua yang dipilih secara acak, kami membuat 2 keturunan
# dengan mengambil pecahan gen dari satu induk dan sisa pecahan dari induk lain
def crossover(pop, crossover_rate):
    offspring = np.zeros((crossover_rate, pop.shape[1]))
    for i in range(int(crossover_rate/2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        cutting_point = np.random.randint(1, pop.shape[1])
        offspring[2*i, 0:cutting_point] = pop[r1, 0:cutting_point]
        offspring[2*i, cutting_point:] = pop[r2, cutting_point:]
        offspring[2*i+1, 0:cutting_point] = pop[r2, 0:cutting_point]
        offspring[2*i+1, cutting_point:] = pop[r1, cutting_point:]

    return offspring    # arr(crossover_size x n_var)

# Pada setiap iterasi, dari 2 orang tua yang dipilih secara acak, kami membuat 2 keturunan
# dengan menukar sejumlah gen/koordinat antar orang tua
def mutation(pop, mutation_rate):
    offspring = np.zeros((mutation_rate, pop.shape[1]))
    for i in range(int(mutation_rate/2)):
        r1 = np.random.randint(0, pop.shape[0])
        r2 = np.random.randint(0, pop.shape[0])
        while r1 == r2:
            r1 = np.random.randint(0, pop.shape[0])
            r2 = np.random.randint(0, pop.shape[0])
        # pilih hanya satu gen/koordinat per kromosom/solusi untuk mutasi di sini.
        # Untuk solusi biner, jumlah gen untuk mutasi dapat berubah-ubah
        cutting_point = np.random.randint(0, pop.shape[1])
        offspring[2*i] = pop[r1]
        offspring[2*i, cutting_point] = pop[r2, cutting_point]
        offspring[2*i+1] = pop[r2]
        offspring[2*i+1, cutting_point] = pop[r1, cutting_point]

    return offspring    # arr(mutation_size x n_var)

# Buat sejumlah keturunan Q dengan menambahkan perpindahan koordinat tetap
# gen/koordinat induk yang dipilih secara acak
def local_search(pop, n_sol, step_size):
    # number of offspring chromosomes generated from the local search
    offspring = np.zeros((n_sol, pop.shape[1]))
    for i in range(n_sol):
        r1 = np.random.randint(0, pop.shape[0])
        chromosome = pop[r1, :]
        r2 = np.random.randint(0, pop.shape[1])
        chromosome[r2] += np.random.uniform(-step_size, step_size)
        if chromosome[r2] < lb[r2]:
            chromosome[r2] = lb[r2]
        if chromosome[r2] > ub[r2]:
            chromosome[r2] = ub[r2]

        offspring[i,:] = chromosome
    return offspring    # arr(loc_search_size x n_var)

# Hitung nilai kebugaran (fungsi obj) untuk setiap kromosom / solusi
# Fungsi Kursawe - https://en.wikipedia.org/wiki/Test_functions_for_optimization
def evaluation(pop):
    fitness_values = np.zeros((pop.shape[0], 2)) # 2 nilai untuk setiap kromosom / solusi
    for i,x in enumerate(pop):
        obj1 = 0
        for j in range(2):
            obj1 += - 10*math.exp(-0.2*math.sqrt((x[j])**2 + (x[j+1])**2))

        obj2 = 0
        for j in range(3):
            obj2 += (abs(x[j]))**0.8 + 5*math.sin((x[j])**3)

        fitness_values[i,0] = obj1
        fitness_values[i,1] = obj2

    return fitness_values   # arr(pop_size x 2)

# Perkirakan seberapa rapat nilai fitness di depan Pareto.
def crowding_calculation(fitness_values):
    pop_size = len(fitness_values[:, 0])
    fitness_value_number = len(fitness_values[0, :])                    # == n of objective functions
    matrix_for_crowding = np.zeros((pop_size, fitness_value_number))    # arr(pop_size x 2)
    normalized_fitness_values = (fitness_values - fitness_values.min(0))/fitness_values.ptp(0)  # arr.ptp(0) array of max elem in each col

    for i in range(fitness_value_number):
        crowding_results = np.zeros(pop_size)
        crowding_results[0] = 1 #titik ekstrim memiliki jarak kerumun maksimal
        crowding_results[pop_size - 1] = 1 # titik ekstrim memiliki jarak kerumun maksimal
        sorted_normalized_fitness_values = np.sort(normalized_fitness_values[:,i])
        sorted_normalized_values_index = np.argsort(normalized_fitness_values[:,i])
        # perhitungan jarak keramaian, Katakan untuk fitness1[i], crowding = fitness1[i+1] - fitness1[i-1]
        crowding_results[1:pop_size - 1] = (sorted_normalized_fitness_values[2:pop_size] - sorted_normalized_fitness_values[0:pop_size - 2])
        re_sorting = np.argsort(sorted_normalized_values_index)
        matrix_for_crowding[:, i] = crowding_results[re_sorting]

    crowding_distance = np.sum(matrix_for_crowding, axis=1) # Pada fitness1 - fitness2 merencanakan, setiap titik di depan pareto memiliki nomor jarak kerumun

    return crowding_distance    # arr(pop_size,)

# Jarak kerumunan digunakan untuk menjaga keragaman solusi di depan Pareto.
# Hapus sejumlah solusi yang menggumpal menjadi banyak
def remove_using_crowding(fitness_values, number_solutions_needed):
    pop_index = np.arange(fitness_values.shape[0])
    crowding_distance = crowding_calculation(fitness_values)
    selected_pop_index = np.zeros(number_solutions_needed)
    selected_fitness_values = np.zeros((number_solutions_needed, len(fitness_values[0, :])))    # arr(num_sol_needed x 2)
    for i in range(number_solutions_needed):
        pop_size = pop_index.shape[0]
        solution_1 = rn.randint(0, pop_size - 1)
        solution_2 = rn.randint(0, pop_size - 1)
        if crowding_distance[solution_1] >= crowding_distance[solution_2]:
            # solusi 1 lebih baik dari solusi 2
            selected_pop_index[i] = pop_index[solution_1]
            selected_fitness_values[i, :] = fitness_values[solution_1, :]
            pop_index = np.delete(pop_index, (solution_1), axis=0)
            fitness_values = np.delete(fitness_values, (solution_1), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_1), axis=0)
        else:
            # solusi 2 lebih baik dari solusi 1
            selected_pop_index[i] = pop_index[solution_2]
            selected_fitness_values[i, :] = fitness_values[solution_2, :]
            pop_index = np.delete(pop_index, (solution_2), axis=0)
            fitness_values = np.delete(fitness_values, (solution_2), axis=0)
            crowding_distance = np.delete(crowding_distance, (solution_2), axis=0)

    selected_pop_index = np.asarray(selected_pop_index, dtype=int)

    return selected_pop_index   # arr(n_sol_needed,)

# temukan indeks solusi yang mendominasi yang lain
def pareto_front_finding(fitness_values, pop_index):
    pop_size = fitness_values.shape[0]
    pareto_front = np.ones(pop_size, dtype=bool)    # semua bernilai True
    for i in range(pop_size):
        for j in range(pop_size):
            if all(fitness_values[j] <= fitness_values[i]) and any(fitness_values[j] < fitness_values[i]):
                pareto_front[i] = 0 # i tidak di depan pareto karena j mendominasi i
                break

    return pop_index[pareto_front]  # arr(len_pareto_front,)

# ulangi pemilihan depan Pareto untuk membangun populasi dalam batas ukuran yang ditentukan
def selection(pop, fitness_values, pop_size):

    pop_index_0 = np.arange(pop.shape[0])   # id pop yang tidak dipilih
    pop_index = np.arange(pop.shape[0])     # semua id pop. len = len(pop_size)
    pareto_front_index = []

    while len(pareto_front_index) < pop_size:   # pop_size = initial_pop_size
        new_pareto_front = pareto_front_finding(fitness_values[pop_index_0, :], pop_index_0)
        total_pareto_size = len(pareto_front_index) + len(new_pareto_front)

        # periksa ukuran pareto_front, jika lebih besar dari pop_size maka hapus beberapa
        if total_pareto_size > pop_size:
            number_solutions_needed = pop_size - len(pareto_front_index)
            selected_solutions = remove_using_crowding(fitness_values[new_pareto_front], number_solutions_needed)
            new_pareto_front = new_pareto_front[selected_solutions]

        pareto_front_index = np.hstack((pareto_front_index, new_pareto_front))
        remaining_index = set(pop_index) - set(pareto_front_index)
        pop_index_0 = np.array(list(remaining_index))

    selected_pop = pop[pareto_front_index.astype(int)]

    return selected_pop     # arr(pop_size x n_var)

# Parameter
n_var = 3                   # kromosom memiliki 3 koordinat/gen
lb = [-5, -5, -5]
ub = [5, 5, 5]
pop_size = 150              # jumlah awal kromosom
rate_crossover = 20         # jumlah kromosom yang dterapkan ke crossower
rate_mutation = 20          # jumlah kromosom yang diterapkan untuk mutasi
rate_local_search = 10      # jumlah kromosom yang diterapkan untuk local_search
step_size = 0.1             # koordinat perpindahan selama local_search
maximum_generation = 150    # jumlah iterasi
pop = random_population(n_var, pop_size, lb, ub)    # populasi orang tua awal P
print(pop.shape)


# NSGA-II perulangan utama
for i in range(maximum_generation):
    offspring_from_crossover = crossover(pop, rate_crossover)
    offspring_from_mutation = mutation(pop, rate_mutation)
    offspring_from_local_search = local_search(pop, rate_local_search, step_size)

    # meambahkan anak Q (cross-over, mutasi, pencarian lokal) ke orang tua P
    # memiliki orang tua dalam campuran, yaitu memungkinkan orang tua untuk maju ke iteration - Elitism berikutnya
    pop = np.append(pop, offspring_from_crossover, axis=0)
    pop = np.append(pop, offspring_from_mutation, axis=0)
    pop = np.append(pop, offspring_from_local_search, axis=0)
    # cetak (pop.shape)
    fitness_values = evaluation(pop)
    pop = selection(pop, fitness_values, pop_size)  # mengatur pereto yang diinginkan front size = pop_size
    print('iteration:', i)

# Visualisasi Pareto depan
fitness_values = evaluation(pop)
index = np.arange(pop.shape[0]).astype(int)
pareto_front_index = pareto_front_finding(fitness_values, index)
pop = pop[pareto_front_index, :]
print("_________________")
print("Optimal solutions:")
print("       x1               x2                 x3")
print(pop) # tampilkan solusi optimal
fitness_values = fitness_values[pareto_front_index]
print("______________")
print("Fitness values:")
print("  objective 1    objective 2")
print(fitness_values)
plt.scatter(fitness_values[:, 0],fitness_values[:, 1], label='Pareto optimal front')
plt.legend(loc='best')
plt.xlabel('Objective function F1')
plt.ylabel('Objective function F2')
plt.grid(b=1)
plt.show()

# buat fungsi dekomposisi bernama local_loop
# local_loop akan menghitung setiap bagiannya
# gunakan 4/(1+x^2), perhatikan batas awal dan akhir untuk dekomposisi
# misalkan size = 4 maka proses 0 menghitung 0-25, proses 1 menghitung 26-50, dst
def local_loop(num_steps,begin,end):
    step = 1.0/num_steps
    sum = 0
    for i in range(begin,end):
        x= (i+0.5)*step
        sum = sum + 4.0/(1.0+x*x)
    print (sum)
    return sum

# fungsi Pi
def Pi(num_steps):

    # buat COMM
    comm = MPI.COMM_WORLD

    # dapatkan rank proses
    rank = comm.Get_rank()

    # dapatkan total proses berjalan
    size = comm.Get_size()

    # buat variabel baru yang merupakan num_steps/total proses
    num_steps2 = int(num_steps/size)

    # cari local_sum
    # local_sum merupakan hasil dari memanggil fungsi local_loop
    local_sum = local_loop(num_steps,rank*num_steps2,(rank+1)*num_steps2)

    # lakukan penjumlahan dari local_sum proses-proses yang ada ke proses 0
    # bisa digunakan reduce atau p2p sum
    sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # jika saya proses dengan rank 0  maka tampilkan hasilnya
    if rank == 0:
        pi = sum / num_steps
        print (pi)

# panggil fungsi utama
if __name__ == '__main__':
    Pi(10000)