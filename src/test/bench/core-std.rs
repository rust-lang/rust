// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-lexer-test FIXME #15679
// Microbenchmarks for various functions in std and extra

#![feature(unboxed_closures)]

use std::old_io::File;
use std::iter::repeat;
use std::mem::swap;
use std::env;
use std::rand::Rng;
use std::rand;
use std::str;
use std::time::Duration;
use std::vec;

fn main() {
    let argv: Vec<String> = env::args().collect();

    macro_rules! bench {
        ($id:ident) =>
            (maybe_run_test(&argv,
                            stringify!($id).to_string(),
                            $id))
    }

    bench!(shift_push);
    bench!(read_line);
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
    } else if argv.len() > 0 {
        run_test = argv.iter().any(|x| x == &"all".to_string()) || argv.iter().any(|x| x == &name)
    }

    if !run_test {
        return
    }

    let dur = Duration::span(test);

    println!("{}:\t\t{} ms", name, dur.num_milliseconds());
}

fn shift_push() {
    let mut v1 = repeat(1).take(30000).collect::<Vec<_>>();
    let mut v2 = Vec::new();

    while v1.len() > 0 {
        v2.push(v1.remove(0));
    }
}

fn read_line() {
    use std::old_io::BufferedReader;

    let mut path = Path::new(env!("CFG_SRC_DIR"));
    path.push("src/test/bench/shootout-k-nucleotide.data");

    for _ in 0..3 {
        let mut reader = BufferedReader::new(File::open(&path).unwrap());
        for _line in reader.lines() {
        }
    }
}

fn vec_plus() {
    let mut r = rand::thread_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = repeat(i).take(r.gen_range(0, i + 1)).collect::<Vec<_>>();
        if r.gen() {
            v.extend(rv.into_iter());
        } else {
            let mut rv = rv.clone();
            rv.push_all(&v);
            v = rv;
        }
        i += 1;
    }
}

fn vec_append() {
    let mut r = rand::thread_rng();

    let mut v = Vec::new();
    let mut i = 0;
    while i < 1500 {
        let rv = repeat(i).take(r.gen_range(0, i + 1)).collect::<Vec<_>>();
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
    let mut r = rand::thread_rng();

    let mut v = Vec::new();
    for i in 0..1500 {
        let mut rv = repeat(i).take(r.gen_range(0, i + 1)).collect::<Vec<_>>();
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
