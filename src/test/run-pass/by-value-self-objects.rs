// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static mut destructor_count: uint = 0;

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
        assert!(self.x == 2);
        assert!(self.y == 3);
        assert!(self.z == 4);
        assert!(self.s.as_slice() == "hello");
        assert!(x == 5);
    }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("bye 1!");
        unsafe {
            destructor_count += 1;
        }
    }
}

impl Foo for int {
    fn foo(self, x: int) {
        println!("{}", x * x);
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
    println!("bye 2!");
}

fn g() {
    let s = 2i;
    let st = box s as Box<Foo>;
    st.foo(3);
    println!("bye 3!");
}

fn main() {
    f();

    unsafe {
        assert!(destructor_count == 1);
    }

    g();
}

