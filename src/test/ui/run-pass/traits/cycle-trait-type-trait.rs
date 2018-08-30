// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test a case where a supertrait references a type that references
// the original trait. This poses no problem at the moment.

// pretty-expanded FIXME #23616

trait Chromosome: Get<Struct<i32>> {
}

trait Get<A> {
    fn get(&self) -> A;
}

struct Struct<C:Chromosome> { c: C }

impl Chromosome for i32 { }

impl Get<Struct<i32>> for i32 {
    fn get(&self) -> Struct<i32> {
        Struct { c: *self }
    }
}

fn main() { }
