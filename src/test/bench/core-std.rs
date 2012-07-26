// Microbenchmarks for various functions in core and std

use std;

import std::time::precise_time_s;
import std::map;
import std::map::{map, hashmap};

import io::{reader, reader_util};

fn main(argv: ~[~str]) {
    #macro[
        [#bench[id],
         maybe_run_test(argv, #stringify(id), id)
        ]
    ];

    let tests = vec::view(argv, 1, argv.len());

    #bench[shift_push];
    #bench[read_line];
    #bench[str_set];
    #bench[vec_plus];
    #bench[vec_append];
    #bench[vec_push_all];
}

fn maybe_run_test(argv: &[~str], name: ~str, test: fn()) {
    let mut run_test = false;

    if os::getenv(~"RUST_BENCH").is_some() { run_test = true }
    else if argv.len() > 0 {
        run_test = argv.contains(~"all") || argv.contains(name)
    }

    if !run_test { ret }

    let start = precise_time_s();
    test();
    let stop = precise_time_s();

    io::println(#fmt("%s:\t\t%f ms", name, (stop - start) * 1000f));
}

fn shift_push() {
    let mut v1 = vec::from_elem(30000, 1);
    let mut v2 = ~[];

    while v1.len() > 0 {
        vec::push(v2, vec::shift(v1));
    }
}

fn read_line() {
    let path = path::connect(
        #env("CFG_SRC_DIR"),
        ~"src/test/bench/shootout-k-nucleotide.data"
    );

    for int::range(0, 3) |_i| {
        let reader = result::get(io::file_reader(path));
        while !reader.eof() {
            reader.read_line();
        }
    }
}

fn str_set() {
    let r = rand::rng();

    let s = map::hashmap(str::hash, str::eq);

    for int::range(0, 1000) |_i| {
        map::set_add(s, r.gen_str(10));
    }
    
    let mut found = 0;
    for int::range(0, 1000) |_i| {
        alt s.find(r.gen_str(10)) {
          some(_) { found += 1; }
          none { }
        }
    }
}

fn vec_plus() {
    let r = rand::rng();

    let mut v = ~[]; 
    let mut i = 0;
    while i < 1500 {
        let rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen_bool() {
            v += rv;
        }
        else {
            v = rv + v;
        }
        i += 1;
    }
}

fn vec_append() {
    let r = rand::rng();

    let mut v = ~[];
    let mut i = 0;
    while i < 1500 {
        let rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen_bool() {
            v = vec::append(v, rv);
        }
        else {
            v = vec::append(rv, v);
        }
        i += 1;
    }
}

fn vec_push_all() {
    let r = rand::rng();

    let mut v = ~[];
    for uint::range(0, 1500) |i| {
        let mut rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen_bool() {
            vec::push_all(v, rv);
        }
        else {
            v <-> rv;
            vec::push_all(v, rv);
        }
    }
}
