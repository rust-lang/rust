// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue_30123_aux.rs

extern crate issue_30123_aux;
use issue_30123_aux::*;

fn main() {
    let ug = Graph::<i32, i32>::new_undirected();
    //~^ ERROR no associated item named `new_undirected` found for type
}
