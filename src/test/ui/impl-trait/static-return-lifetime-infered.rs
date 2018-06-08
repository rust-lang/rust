// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A {
    x: [(u32, u32); 10]
}

impl A {
    fn iter_values_anon(&self) -> impl Iterator<Item=u32> {
        self.x.iter().map(|a| a.0)
    }
    //~^^^ ERROR can't infer an appropriate lifetime
    fn iter_values<'a>(&'a self) -> impl Iterator<Item=u32> {
        self.x.iter().map(|a| a.0)
    }
    //~^^^ ERROR can't infer an appropriate lifetime
}

fn main() {}
