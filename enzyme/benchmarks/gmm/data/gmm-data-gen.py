import os
import sys
import numpy as np

# function printing to stderr
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_points_dir_name(n):
    if n == 1000:
        return "1k"
    if n == 10000:
        return "10k"
    if n == 2500000:
        return "2.5M"
    raise ValueError("Undefined number of points: {n}")

def replicate_point(n):
    if n == 1000:
        return False
    if n == 10000:
        return False
    if n == 2500000:
        return True
    raise ValueError("Undefined number of points: {n}")

def generate(data_uniform, data_normal, D, k, n):

    gamma = 1.
    m = 0

    view_uniform = data_uniform[:]
    view_normal = data_normal[:]

    filename = f"gmm_d{D}_K{k}.txt"

    # write data to file
    with open(os.path.join(os.getcwd(), f"data/gmm/{get_points_dir_name(n)}", filename), 'w') as outfile:

        outfile.write(f"{D} {k} {n}\n")

        # alpha
        for i in range(k):
            outfile.write(f"{view_normal[0]:.6f}\n")
            view_normal = view_normal[1:]

        # mu
        for i in range(k):
            for j in range(D):
                outfile.write(f"{view_uniform[0]:.6f} ")
                view_uniform = view_uniform[1:]
            outfile.write("\n")

        # q
        for i in range(k):
            for j in range(D + D*(D-1)//2):
                outfile.write(f"{view_normal[0]:.6f} ")
                view_normal = view_normal[1:]
            outfile.write("\n")

        # x
        if replicate_point(n):
            for j in range(D):
                outfile.write(f"{view_normal[0]:.6f} ")
                view_normal = view_normal[1:]
            outfile.write("\n")
        else:
            for i in range(n):
                for j in range(D):
                    outfile.write(f"{view_normal[0]:.6f} ")
                    view_normal = view_normal[1:]
                outfile.write("\n")

        outfile.write(f"{gamma:.6f} {m} \n")



def generator(d, K, N):

    # uniform distribution parameters
    low = 0
    high = 1
    amount_of_uniform_numbers = K[-1]*d
    data_uniform = np.random.uniform(low, high, amount_of_uniform_numbers)

    # normal distribution parameters
    mean = 0
    sigma = 1
    amount_of_normal_numbers = K[-1]*(1 + d + d*(d-1)//2) + N[-2]*d
    data_normal = np.random.normal(mean, sigma, amount_of_normal_numbers)

    for n in N:
        for k in K:
            generate(data_uniform, data_normal, d, k, n)

def main(argv):
    try:

        d = 128
        K = [5, 10, 25, 50, 100, 200]
        N = [1000, 10000, 2500000]

        # generate GMM models with d dimensions
        generator(d, K, N)

    except RuntimeError as ex:
        eprint("Runtime exception caught: ", ex)
    except Exception as ex:
        eprint("An exception caught: ", ex)

    return 0

if __name__ == "__main__":
    main(sys.argv[:])