import math

import numpy as np

# Data size variables
layer_counts = [2, 4]
char_counts = [1024, 4096]
text_input = "lstm_full.txt"


# Read from the full text file
def read_full_text(fn, char_count):
    full_text_file = open(text_input, encoding="utf8")
    full_text = full_text_file.read(char_count)
    full_text_file.close()

    return full_text


def get_char_bits(text):
    return math.ceil(math.log2(max([ord(c) for c in text])))


def text_to_matrix(text, bits):
    # bits = math.ceil(math.log2(max([ord(c) for c in text])))
    return np.array(list(map(lambda c: list(map(lambda b: int(b), bin(ord(c))[2:].zfill(bits))), text)))


def f_write_mat(fid, matrix):
    for row in matrix:
        fid.write(" ".join([str(n) for n in row]))
        fid.write("\n")
    fid.write("\n")


# Read in all text data
full_text = read_full_text(text_input, max(char_counts))

# Loop through each size
for layer_count in layer_counts:
    for char_count in char_counts:
        # Get text extract
        use_text = full_text[: char_count]
        char_bits = get_char_bits(use_text)
        text_mat = text_to_matrix(use_text, char_bits)

        # Randomly generate past state, and parameters
        state = np.random.random((2 * layer_count, char_bits))
        main_params = np.random.random((2 * layer_count, char_bits * 4))
        extra_params = np.random.random((3, char_bits))

        # Write to file
        f = open(f"lstm_l{layer_count}_c{char_count}.txt", "w")
        f.write(f"{layer_count} {char_count} {char_bits}\n\n")
        f_write_mat(f, main_params)
        f_write_mat(f, extra_params)
        f_write_mat(f, state)
        f_write_mat(f, text_mat)
        f.write("\n")
        f.close()
