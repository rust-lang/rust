// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo { x: int }

impl Foo {
    fn stuff(&mut self) -> &self/mut Foo {
        return self;
    }
}

fn main() {
    let mut x = @mut Foo { x: 3 };
    // Neither of the next two lines should cause an error
    let _ = x.stuff(); 
    x.stuff();
}