// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
// error-pattern:complex called
// error-pattern:dropped
// error-pattern:exit

use std::io::{self, Write};

struct Droppable;
impl Drop for Droppable {
    fn drop(&mut self) {
        write!(io::stderr(), "dropped\n");
    }
}

// return value of this function is copied into the return slot
fn complex() -> u64 {
    write!(io::stderr(), "complex called\n");
    42
}


#[rustc_mir]
fn mir() -> u64 {
    let x = Droppable;
    return complex();
    drop(x);
}

pub fn main() {
    assert_eq!(mir(), 42);
    panic!("exit");
}
