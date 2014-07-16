// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    x: int,
    y: int,
}

impl Index<String,int> for Foo {
    fn index<'a>(&'a self, z: &String) -> &'a int {
        if z.as_slice() == "x" {
            &self.x
        } else {
            &self.y
        }
    }
}

impl IndexMut<String,int> for Foo {
    fn index_mut<'a>(&'a mut self, z: &String) -> &'a mut int {
        if z.as_slice() == "x" {
            &mut self.x
        } else {
            &mut self.y
        }
    }
}

struct Bar {
    x: int,
}

impl Index<int,int> for Bar {
    fn index<'a>(&'a self, z: &int) -> &'a int {
        &self.x
    }
}

fn main() {
    let mut f = Foo {
        x: 1,
        y: 2,
    };
    let mut s = "hello".to_string();
    let rs = &mut s;
    println!("{}", f[s]);
    //~^ ERROR cannot borrow `s` as immutable because it is also borrowed as mutable
    f[s] = 10;
    //~^ ERROR cannot borrow `s` as immutable because it is also borrowed as mutable
    let s = Bar {
        x: 1,
    };
    s[2] = 20;
    //~^ ERROR cannot assign to immutable dereference (dereference is implicit, due to indexing)
}


