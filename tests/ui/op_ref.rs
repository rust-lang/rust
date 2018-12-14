// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variables, clippy::blacklisted_name)]

use std::collections::HashSet;

fn main() {
    let tracked_fds: HashSet<i32> = HashSet::new();
    let new_fds = HashSet::new();
    let unwanted = &tracked_fds - &new_fds;

    let foo = &5 - &6;

    let bar = String::new();
    let bar = "foo" == &bar;

    let a = "a".to_string();
    let b = "a";

    if b < &a {
        println!("OK");
    }
}
