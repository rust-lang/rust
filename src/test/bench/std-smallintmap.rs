// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Microbenchmark for the smallintmap library

extern crate collections;
extern crate time;

use collections::SmallIntMap;
use std::os;
use std::uint;

fn append_sequential(min: uint, max: uint, map: &mut SmallIntMap<uint>) {
    for i in range(min, max) {
        map.insert(i, i + 22u);
    }
}

fn check_sequential(min: uint, max: uint, map: &SmallIntMap<uint>) {
    for i in range(min, max) {
        assert_eq!(*map.get(&i), i + 22u);
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_owned(), "100000".to_owned(), "100".to_owned())
    } else if args.len() <= 1u {
        vec!("".to_owned(), "10000".to_owned(), "50".to_owned())
    } else {
        args.move_iter().collect()
    };
    let max = from_str::<uint>(args.get(1).as_slice()).unwrap();
    let rep = from_str::<uint>(args.get(2).as_slice()).unwrap();

    let mut checkf = 0.0;
    let mut appendf = 0.0;

    for _ in range(0u, rep) {
        let mut map = SmallIntMap::new();
        let start = time::precise_time_s();
        append_sequential(0u, max, &mut map);
        let mid = time::precise_time_s();
        check_sequential(0u, max, &map);
        let end = time::precise_time_s();

        checkf += (end - mid) as f64;
        appendf += (mid - start) as f64;
    }

    let maxf = max as f64;

    println!("insert(): {:?} seconds\n", checkf);
    println!("        : {} op/sec\n", maxf/checkf);
    println!("get()   : {:?} seconds\n", appendf);
    println!("        : {} op/sec\n", maxf/appendf);
}
