// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S {
    x: int,
}

impl Drop for S {
    fn drop(&mut self) {}
}

impl S {
    pub fn foo(self) -> int {
        self.bar();
        return self.x;  //~ ERROR use of partially moved value: `self.x`
    }

    pub fn bar(self) {}
}

fn main() {
    let x = S { x: 1 };
    println!("{}", x.foo());
}
