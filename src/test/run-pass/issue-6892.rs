// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensures that destructors are run for expressions of the form "let _ = e;"
// where `e` is a type which requires a destructor.

struct Foo;
struct Bar { x: int }
struct Baz(int);
enum FooBar { _Foo(Foo), _Bar(uint) }

static mut NUM_DROPS: uint = 0;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for Bar {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for Baz {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for FooBar {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

fn main() {
    assert_eq!(unsafe { NUM_DROPS }, 0);
    { let _x = Foo; }
    assert_eq!(unsafe { NUM_DROPS }, 1);
    { let _x = Bar { x: 21 }; }
    assert_eq!(unsafe { NUM_DROPS }, 2);
    { let _x = Baz(21); }
    assert_eq!(unsafe { NUM_DROPS }, 3);
    { let _x = _Foo(Foo); }
    assert_eq!(unsafe { NUM_DROPS }, 5);
    { let _x = _Bar(42u); }
    assert_eq!(unsafe { NUM_DROPS }, 6);

    { let _ = Foo; }
    assert_eq!(unsafe { NUM_DROPS }, 7);
    { let _ = Bar { x: 21 }; }
    assert_eq!(unsafe { NUM_DROPS }, 8);
    { let _ = Baz(21); }
    assert_eq!(unsafe { NUM_DROPS }, 9);
    { let _ = _Foo(Foo); }
    assert_eq!(unsafe { NUM_DROPS }, 11);
    { let _ = _Bar(42u); }
    assert_eq!(unsafe { NUM_DROPS }, 12);
}
