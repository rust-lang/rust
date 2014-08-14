// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unused_variable)]
#![allow(non_camel_case_types)]
#![deny(dead_code)]

#![crate_type="lib"]

extern crate libc;

pub use extern_foo as x;
extern {
    fn extern_foo();
}

struct Foo; //~ ERROR: code is never used
impl Foo {
    fn foo(&self) { //~ ERROR: code is never used
        bar()
    }
}

fn bar() { //~ ERROR: code is never used
    fn baz() {} //~ ERROR: code is never used

    Foo.foo();
    baz();
}

// no warning
struct Foo2;
impl Foo2 { fn foo2(&self) { bar2() } }
fn bar2() {
    fn baz2() {}

    Foo2.foo2();
    baz2();
}

pub fn pub_fn() {
    let foo2_struct = Foo2;
    foo2_struct.foo2();

    blah::baz();
}

mod blah {
    use libc::size_t;
    // not warned because it's used in the parameter of `free` and return of
    // `malloc` below, which are also used.
    enum c_void {}

    extern {
        fn free(p: *const c_void);
        fn malloc(size: size_t) -> *const c_void;
    }

    pub fn baz() {
        unsafe { free(malloc(4)); }
    }
}

enum c_void {} //~ ERROR: code is never used
extern {
    fn free(p: *const c_void); //~ ERROR: code is never used
}

// Check provided method
mod inner {
    pub trait Trait {
        fn f(&self) { f(); }
    }

    impl Trait for int {}

    fn f() {}
}

pub fn foo() {
    let a = &1i as &inner::Trait;
    a.f();
}
