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

use std::collections::VecMap;
use std::os;
use std::time::Duration;

fn append_sequential(min: uint, max: uint, map: &mut VecMap<uint>) {
    for i in range(min, max) {
        map.insert(i, i + 22u);
    }
}

fn check_sequential(min: uint, max: uint, map: &VecMap<uint>) {
    for i in range(min, max) {
        assert_eq!(map[i], i + 22u);
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv("RUST_BENCH").is_some() {
        vec!("".to_string(), "100000".to_string(), "100".to_string())
    } else if args.len() <= 1u {
        vec!("".to_string(), "10000".to_string(), "50".to_string())
    } else {
        args.into_iter().collect()
    };
    let max = args[1].parse::<uint>().unwrap();
    let rep = args[2].parse::<uint>().unwrap();

    let mut checkf = Duration::seconds(0);
    let mut appendf = Duration::seconds(0);

    for _ in range(0u, rep) {
        let mut map = VecMap::new();
        let d1 = Duration::span(|| append_sequential(0u, max, &mut map));
        let d2 = Duration::span(|| check_sequential(0u, max, &map));

        checkf = checkf + d2;
        appendf = appendf + d1;
    }

    let maxf = max as f64;

    println!("insert(): {} seconds\n", checkf);
    println!("        : {} op/ms\n", maxf / checkf.num_milliseconds() as f64);
    println!("get()   : {} seconds\n", appendf);
    println!("        : {} op/ms\n", maxf / appendf.num_milliseconds() as f64);
}
