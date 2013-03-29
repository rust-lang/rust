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

extern mod std;
use std::smallintmap::SmallIntMap;
use core::io::WriterUtil;

fn append_sequential(min: uint, max: uint, map: &mut SmallIntMap<uint>) {
    for uint::range(min, max) |i| {
        map.insert(i, i + 22u);
    }
}

fn check_sequential(min: uint, max: uint, map: &SmallIntMap<uint>) {
    for uint::range(min, max) |i| {
        assert!(*map.get(&i) == i + 22u);
    }
}

fn main() {
    let args = os::args();
    let args = if os::getenv(~"RUST_BENCH").is_some() {
        ~[~"", ~"100000", ~"100"]
    } else if args.len() <= 1u {
        ~[~"", ~"10000", ~"50"]
    } else {
        args
    };
    let max = uint::from_str(args[1]).get();
    let rep = uint::from_str(args[2]).get();

    let mut checkf = 0.0;
    let mut appendf = 0.0;

    for uint::range(0u, rep) |_r| {
        let mut map = SmallIntMap::new();
        let start = std::time::precise_time_s();
        append_sequential(0u, max, &mut map);
        let mid = std::time::precise_time_s();
        check_sequential(0u, max, &map);
        let end = std::time::precise_time_s();

        checkf += (end - mid) as float;
        appendf += (mid - start) as float;
    }

    let maxf = max as float;

    io::stdout().write_str(fmt!("insert(): %? seconds\n", checkf));
    io::stdout().write_str(fmt!("        : %f op/sec\n", maxf/checkf));
    io::stdout().write_str(fmt!("get()   : %? seconds\n", appendf));
    io::stdout().write_str(fmt!("        : %f op/sec\n", maxf/appendf));
}
