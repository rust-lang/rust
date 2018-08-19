// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks to make sure that `dyn Trait + Send` and `dyn Trait + Send + Send` are the same type.
// Issue: #47010

struct Struct;
impl Trait for Struct {}
trait Trait {}

type Send1 = Trait + Send;
type Send2 = Trait + Send + Send;

fn main () {}

impl Trait + Send {
    fn test(&self) { println!("one"); } //~ ERROR duplicate definitions with name `test`
}

impl Trait + Send + Send {
    fn test(&self) { println!("two"); }
}
