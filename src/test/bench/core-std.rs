// Microbenchmarks for various functions in core and std

use std;

import std::time::precise_time_s;

import io::{reader, reader_util};

fn main() {
    #macro[
        [#bench[id],
         run_test(#stringify(id), id)]
    ];

    #bench[shift_push];
    #bench[read_line];
}

fn run_test(name: str, test: fn()) {
    let start = precise_time_s();
    test();
    let stop = precise_time_s();

    io::println(#fmt("%s:\t\t%f ms", name, (stop - start) * 1000f));
}

fn shift_push() {
    let mut v1 = vec::from_elem(30000, 1);
    let mut v2 = []/~;

    while v1.len() > 0 {
        vec::push(v2, vec::shift(v1));
    }
}

fn read_line() {
    let path = path::connect(
        #env("CFG_SRC_DIR"),
        "src/test/bench/shootout-k-nucleotide.data"
    );

    for int::range(0, 3) {|_i|
        let reader = result::get(io::file_reader(path));
        while !reader.eof() {
            reader.read_line();
        }
    }
}
