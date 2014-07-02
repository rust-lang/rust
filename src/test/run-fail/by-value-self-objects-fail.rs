// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:explicit failure

trait Foo {
    fn foo(self, x: int);
}

struct S {
    x: int,
    y: int,
    z: int,
    s: String,
}

impl Foo for S {
    fn foo(self, x: int) {
        fail!()
    }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("bye 1!");
    }
}

fn f() {
    let s = S {
        x: 2,
        y: 3,
        z: 4,
        s: "hello".to_string(),
    };
    let st = box s as Box<Foo>;
    st.foo(5);
}

fn main() {
    f();
}


