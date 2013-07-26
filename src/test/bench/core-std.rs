// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Microbenchmarks for various functions in std and extra

extern mod extra;

use extra::time::precise_time_s;
use std::io;
use std::os;
use std::rand::RngUtil;
use std::rand;
use std::str;
use std::util;
use std::vec;

macro_rules! bench (
    ($argv:expr, $id:ident) => (maybe_run_test($argv, stringify!($id).to_owned(), $id))
)

fn main() {
    let argv = os::args();
    let _tests = argv.slice(1, argv.len());

    bench!(argv, shift_push);
    bench!(argv, read_line);
    bench!(argv, vec_plus);
    bench!(argv, vec_append);
    bench!(argv, vec_push_all);
    bench!(argv, is_utf8_ascii);
    bench!(argv, is_utf8_multibyte);
}

fn maybe_run_test(argv: &[~str], name: ~str, test: &fn()) {
    let mut run_test = false;

    if os::getenv("RUST_BENCH").is_some() {
        run_test = true
    } else if argv.len() > 0 {
        run_test = argv.iter().any(|x| x == &~"all") || argv.iter().any(|x| x == &name)
    }

    if !run_test {
        return
    }

    let start = precise_time_s();
    test();
    let stop = precise_time_s();

    printfln!("%s:\t\t%f ms", name, (stop - start) * 1000f);
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

    for _ in range(0, 3) {
        let reader = io::file_reader(&path).unwrap();
        while !reader.eof() {
            reader.read_line();
        }
    }
}

fn vec_plus() {
    let mut r = rand::rng();

    let mut v = ~[];
    let mut i = 0;
    while i < 1500 {
        let rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen() {
            v.push_all_move(rv);
        } else {
            v = rv + v;
        }
        i += 1;
    }
}

fn vec_append() {
    let mut r = rand::rng();

    let mut v = ~[];
    let mut i = 0;
    while i < 1500 {
        let rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen() {
            v = vec::append(v, rv);
        }
        else {
            v = vec::append(rv, v);
        }
        i += 1;
    }
}

fn vec_push_all() {
    let mut r = rand::rng();

    let mut v = ~[];
    for i in range(0u, 1500) {
        let mut rv = vec::from_elem(r.gen_uint_range(0, i + 1), i);
        if r.gen() {
            v.push_all(rv);
        }
        else {
            util::swap(&mut v, &mut rv);
            v.push_all(rv);
        }
    }
}

fn is_utf8_ascii() {
    let mut v : ~[u8] = ~[];
    for _ in range(0u, 20000) {
        v.push('b' as u8);
        if !str::is_utf8(v) {
            fail!("is_utf8 failed");
        }
    }
}

fn is_utf8_multibyte() {
    let s = "b¢€𤭢";
    let mut v : ~[u8]= ~[];
    for _ in range(0u, 5000) {
        v.push_all(s.as_bytes());
        if !str::is_utf8(v) {
            fail!("is_utf8 failed");
        }
    }
}
