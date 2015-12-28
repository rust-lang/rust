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

#![feature(rand, vec_push_all, time2)]

use std::mem::swap;
use std::env;
use std::__rand::{thread_rng, Rng};
use std::str;
use std::time::Instant;

fn main() {
    let argv: Vec<String> = env::args().collect();

    macro_rules! bench {
        ($id:ident) =>
            (maybe_run_test(&argv,
                            stringify!($id).to_string(),
                            $id))
    }

    bench!(shift_push);
    bench!(vec_plus);
    bench!(vec_append);
    bench!(vec_push_all);
    bench!(is_utf8_ascii);
    bench!(is_utf8_multibyte);
}

fn maybe_run_test<F>(argv: &[String], name: String, test: F) where F: FnOnce() {
    let mut run_test = false;

    if env::var_os("RUST_BENCH").is_some() {
        run_test = true
    } else if !argv.is_empty() {
        run_test = argv.iter().any(|x| x == &"all".to_string()) || argv.iter().any(|x| x == &name)
    }

    if !run_test {
        return
    }

    let start = Instant::now();
    test();
    let dur = start.elapsed();

    println!("{}:\t\t{:?}", name, dur);
}

fn shift_push() {
    let mut v1 = vec![1; 30000];
    let mut v2 = Vec::new();

    while !v1.is_empty() {
        v2.push(v1.remove(0));
    }
}

fn vec_plus() {
    let mut r = thread_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = vec![i; r.gen_range(0, i + 1)];
        if r.gen() {
            v.extend(rv);
        } else {
            let mut rv = rv.clone();
            rv.push_all(&v);
            v = rv;
        }
        i += 1;
    }
}

fn vec_append() {
    let mut r = thread_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = vec![i; r.gen_range(0, i + 1)];
        if r.gen() {
            let mut t = v.clone();
            t.push_all(&rv);
            v = t;
        }
        else {
            let mut t = rv.clone();
            t.push_all(&v);
            v = t;
        }
        i += 1;
    }
}

fn vec_push_all() {
    let mut r = thread_rng();

    let mut v = Vec::new();
    for i in 0..1500 {
        let mut rv = vec![i; r.gen_range(0, i + 1)];
        if r.gen() {
            v.push_all(&rv);
        }
        else {
            swap(&mut v, &mut rv);
            v.push_all(&rv);
        }
    }
}

fn is_utf8_ascii() {
    let mut v : Vec<u8> = Vec::new();
    for _ in 0..20000 {
        v.push('b' as u8);
        if str::from_utf8(&v).is_err() {
            panic!("from_utf8 panicked");
        }
    }
}

fn is_utf8_multibyte() {
    let s = "b¢€𤭢";
    let mut v : Vec<u8> = Vec::new();
    for _ in 0..5000 {
        v.push_all(s.as_bytes());
        if str::from_utf8(&v).is_err() {
            panic!("from_utf8 panicked");
        }
    }
}
