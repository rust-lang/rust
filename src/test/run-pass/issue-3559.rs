// xfail-test #4276

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// rustc --test map_to_str.rs && ./map_to_str
extern mod std;

use core::io::{WriterUtil};

fn check_strs(actual: &str, expected: &str) -> bool
{
    if actual != expected
    {
        io::stderr().write_line(fmt!("Found %s, but expected %s", actual, expected));
        return false;
    }
    return true;
}

fn tester()
{
    let mut table = core::hashmap::HashMap::new();
    table.insert(@~"one", 1);
    table.insert(@~"two", 2);
    assert!(check_strs(table.to_str(), ~"xxx"));   // not sure what expected should be
}

pub fn main() {}
