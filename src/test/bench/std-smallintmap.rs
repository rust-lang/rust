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

extern mod extra;

use extra::smallintmap::SmallIntMap;
use std::io::WriterUtil;
use std::io;
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
        ~[~"", ~"100000", ~"100"]
    } else if args.len() <= 1u {
        ~[~"", ~"10000", ~"50"]
    } else {
        args
    };
    let max = from_str::<uint>(args[1]).unwrap();
    let rep = from_str::<uint>(args[2]).unwrap();

    let mut checkf = 0.0;
    let mut appendf = 0.0;

    for _ in range(0u, rep) {
        let mut map = SmallIntMap::new();
        let start = extra::time::precise_time_s();
        append_sequential(0u, max, &mut map);
        let mid = extra::time::precise_time_s();
        check_sequential(0u, max, &map);
        let end = extra::time::precise_time_s();

        checkf += (end - mid) as f64;
        appendf += (mid - start) as f64;
    }

    let maxf = max as f64;

    io::stdout().write_str(format!("insert(): {:?} seconds\n", checkf));
    io::stdout().write_str(format!("        : {} op/sec\n", maxf/checkf));
    io::stdout().write_str(format!("get()   : {:?} seconds\n", appendf));
    io::stdout().write_str(format!("        : {} op/sec\n", maxf/appendf));
}
