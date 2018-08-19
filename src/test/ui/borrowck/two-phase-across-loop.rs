// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a borrow which starts as a 2-phase borrow and gets
// carried around a loop winds up conflicting with itself.

#![feature(nll)]

struct Foo { x: String }

impl Foo {
    fn get_string(&mut self) -> &str {
        &self.x
    }
}

fn main() {
    let mut foo = Foo { x: format!("Hello, world") };
    let mut strings = vec![];

    loop {
        strings.push(foo.get_string()); //~ ERROR cannot borrow `foo` as mutable
        if strings.len() > 2 { break; }
    }

    println!("{:?}", strings);
}
