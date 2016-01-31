// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(rustc_attrs)]
// error-pattern:unwind happens
// error-pattern:drop 3
// error-pattern:drop 2
// error-pattern:drop 1
use std::io::{self, Write};


/// Structure which will not allow to be dropped twice.
struct Droppable(bool, u32);
impl Drop for Droppable {
    fn drop(&mut self) {
        if self.0 {
            writeln!(io::stderr(), "{} dropped twice", self.1);
            ::std::process::exit(1);
        }
        writeln!(io::stderr(), "drop {}", self.1);
        self.0 = true;
    }
}

fn may_panic() -> Droppable {
    panic!("unwind happens");
}

#[rustc_mir]
fn mir(d: Droppable){
    let y = Droppable(false, 2);
    let x = [Droppable(false, 1), y, d, may_panic()];
}

fn main() {
    mir(Droppable(false, 3));
}
