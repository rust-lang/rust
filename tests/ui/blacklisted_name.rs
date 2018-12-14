// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(
    dead_code,
    clippy::similar_names,
    clippy::single_match,
    clippy::toplevel_ref_arg,
    unused_mut,
    unused_variables
)]
#![warn(clippy::blacklisted_name)]

fn test(foo: ()) {}

fn main() {
    let foo = 42;
    let bar = 42;
    let baz = 42;

    let barb = 42;
    let barbaric = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(bar), baz @ Some(_)) => (),
        _ => (),
    }
}

fn issue_1647(mut foo: u8) {
    let mut bar = 0;
    if let Some(mut baz) = Some(42) {}
}

fn issue_1647_ref() {
    let ref bar = 0;
    if let Some(ref baz) = Some(42) {}
}

fn issue_1647_ref_mut() {
    let ref mut bar = 0;
    if let Some(ref mut baz) = Some(42) {}
}
