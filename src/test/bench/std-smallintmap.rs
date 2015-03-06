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

#![feature(collections, std_misc)]

use std::collections::VecMap;
use std::env;
use std::time::Duration;

fn append_sequential(min: usize, max: usize, map: &mut VecMap<usize>) {
    for i in min..max {
        map.insert(i, i + 22);
    }
}

fn check_sequential(min: usize, max: usize, map: &VecMap<usize>) {
    for i in min..max {
        assert_eq!(map[i], i + 22);
    }
}

fn main() {
    let args = env::args();
    let args = if env::var_os("RUST_BENCH").is_some() {
        vec!("".to_string(), "100000".to_string(), "100".to_string())
    } else if args.len() <= 1 {
        vec!("".to_string(), "10000".to_string(), "50".to_string())
    } else {
        args.collect()
    };
    let max = args[1].parse::<usize>().unwrap();
    let rep = args[2].parse::<usize>().unwrap();

    let mut checkf = Duration::seconds(0);
    let mut appendf = Duration::seconds(0);

    for _ in 0..rep {
        let mut map = VecMap::new();
        let d1 = Duration::span(|| append_sequential(0, max, &mut map));
        let d2 = Duration::span(|| check_sequential(0, max, &map));

        checkf = checkf + d2;
        appendf = appendf + d1;
    }

    let maxf = max as f64;

    println!("insert(): {} seconds\n", checkf);
    println!("        : {} op/ms\n", maxf / checkf.num_milliseconds() as f64);
    println!("get()   : {} seconds\n", appendf);
    println!("        : {} op/ms\n", maxf / appendf.num_milliseconds() as f64);
}
