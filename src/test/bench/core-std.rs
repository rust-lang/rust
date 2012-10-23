// xfail-pretty
// Microbenchmarks for various functions in core and std

extern mod std;

use std::time::precise_time_s;
use std::map;
use std::map::{Map, HashMap};

use io::{Reader, ReaderUtil};

fn main() {
    let argv = os::args();
    #macro[
        [#bench[id],
         maybe_run_test(argv, #stringify(id), id)
        ]
    ];

    let tests = vec::view(argv, 1, argv.len());

    bench!(shift_push);
    bench!(read_line);
    bench!(str_set);
    bench!(vec_plus);
    bench!(vec_append);
    bench!(vec_push_all);
}

fn maybe_run_test(argv: &[~str], name: ~str, test: fn()) {
    let mut run_test = false;

    if os::getenv(~"RUST_BENCH").is_some() { run_test = true }
    else if argv.len() > 0 {
        run_test = argv.contains(&~"all") || argv.contains(&name)
    }

    if !run_test { return }

    let start = precise_time_s();
    test();
    let stop = precise_time_s();

    io::println(fmt!("%s:\t\t%f ms", name, (stop - start) * 1000f));
}

fn shift_push() {
    let mut v1 = vec::from_elem(30000, 1);
    let mut v2 = ~[];

    while v1.len() > 0 {
        v2.push(v1.shift());
    }
}

fn read_line() {
    let path = Path(env!("CFG_SRC_DIR"))
        .push_rel(&Path("src/test/bench/shootout-k-nucleotide.data"));

    for int::range(0, 3) |_i| {
        let reader = result::get(&io::file_reader(&path));
        while !reader.eof() {
            reader.read_line();
        }
    }
}

fn str_set() {
    let r = rand::Rng();

    let s = map::HashMap();

    for int::range(0, 1000) |_i| {
        map::set_add(s, r.gen_str(10));
    }
    
    let mut found = 0;
    for int::range(0, 1000) |_i| {
        match s.find(r.gen_str(10)) {
          Some(_) => { found += 1; }
          None => { }
        }
    }
}

fn vec_plus() {
    let r = rand::Rng();

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
    let r = rand::Rng();

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
    let r = rand::Rng();

    let mut v = ~[];
    for uint::range(0, 1500) |i| {
        let mut rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen_bool() {
            v.push_all(rv);
        }
        else {
            v <-> rv;
            v.push_all(rv);
        }
    }
}
