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

#![feature(macro_rules)]

extern crate time;

use time::precise_time_s;
use std::rand;
use std::rand::Rng;
use std::mem::swap;
use std::os;
use std::str;
use std::vec;
use std::io::File;

fn main() {
    let argv = os::args();
    let _tests = argv.slice(1, argv.len());

    macro_rules! bench (
        ($id:ident) =>
            (maybe_run_test(argv.as_slice(),
                            stringify!($id).to_string(),
                            $id)))

    bench!(shift_push);
    bench!(read_line);
    bench!(vec_plus);
    bench!(vec_append);
    bench!(vec_push_all);
    bench!(is_utf8_ascii);
    bench!(is_utf8_multibyte);
}

fn maybe_run_test(argv: &[String], name: String, test: ||) {
    let mut run_test = false;

    if os::getenv("RUST_BENCH").is_some() {
        run_test = true
    } else if argv.len() > 0 {
        run_test = argv.iter().any(|x| x == &"all".to_string()) || argv.iter().any(|x| x == &name)
    }

    if !run_test {
        return
    }

    let start = precise_time_s();
    test();
    let stop = precise_time_s();

    println!("{}:\t\t{} ms", name, (stop - start) * 1000.0);
}

fn shift_push() {
    let mut v1 = Vec::from_elem(30000, 1i);
    let mut v2 = Vec::new();

    while v1.len() > 0 {
        v2.push(v1.shift().unwrap());
    }
}

fn read_line() {
    use std::io::BufferedReader;

    let mut path = Path::new(env!("CFG_SRC_DIR"));
    path.push("src/test/bench/shootout-k-nucleotide.data");

    for _ in range(0u, 3) {
        let mut reader = BufferedReader::new(File::open(&path).unwrap());
        for _line in reader.lines() {
        }
    }
}

fn vec_plus() {
    let mut r = rand::task_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = Vec::from_elem(r.gen_range(0u, i + 1), i);
        if r.gen() {
            v.push_all_move(rv);
        } else {
            v = rv.clone().append(v.as_slice());
        }
        i += 1;
    }
}

fn vec_append() {
    let mut r = rand::task_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = Vec::from_elem(r.gen_range(0u, i + 1), i);
        if r.gen() {
            v = v.clone().append(rv.as_slice());
        }
        else {
            v = rv.clone().append(v.as_slice());
        }
        i += 1;
    }
}

fn vec_push_all() {
    let mut r = rand::task_rng();

    let mut v = Vec::new();
    for i in range(0u, 1500) {
        let mut rv = Vec::from_elem(r.gen_range(0u, i + 1), i);
        if r.gen() {
            v.push_all(rv.as_slice());
        }
        else {
            swap(&mut v, &mut rv);
            v.push_all(rv.as_slice());
        }
    }
}

fn is_utf8_ascii() {
    let mut v : Vec<u8> = Vec::new();
    for _ in range(0u, 20000) {
        v.push('b' as u8);
        if !str::is_utf8(v.as_slice()) {
            fail!("is_utf8 failed");
        }
    }
}

fn is_utf8_multibyte() {
    let s = "b¢€𤭢";
    let mut v : Vec<u8> = Vec::new();
    for _ in range(0u, 5000) {
        v.push_all(s.as_bytes());
        if !str::is_utf8(v.as_slice()) {
            fail!("is_utf8 failed");
        }
    }
}
