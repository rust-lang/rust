// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test DST raw pointers

trait Trait {
    fn foo(&self) -> int;
}

struct A {
    f: int
}
impl Trait for A {
    fn foo(&self) -> int {
        self.f
    }
}

struct Foo<Sized? T> {
    f: T
}

pub fn main() {
    // raw trait object
    let x = A { f: 42 };
    let z: *const Trait = &x;
    let r = unsafe {
        (&*z).foo()
    };
    assert!(r == 42);

    // raw DST struct
    let p = Foo {f: A { f: 42 }};
    let o: *const Foo<Trait> = &p;
    let r = unsafe {
        (&*o).f.foo()
    };
    assert!(r == 42);

    // raw slice
    let a: *const [_] = &[1i, 2, 3];
    unsafe {
        let b = (*a)[2];
        assert!(b == 3);
        let len = (*a).len();
        assert!(len == 3);
    }

    // raw slice with explicit cast
    let a = &[1i, 2, 3] as *const [_];
    unsafe {
        let b = (*a)[2];
        assert!(b == 3);
        let len = (*a).len();
        assert!(len == 3);
    }

    // raw DST struct with slice
    let c: *const Foo<[_]> = &Foo {f: [1i, 2, 3]};
    unsafe {
        let b = (&*c).f[0];
        assert!(b == 1);
        let len = (&*c).f.len();
        assert!(len == 3);
    }

    // all of the above with *mut
    let mut x = A { f: 42 };
    let z: *mut Trait = &mut x;
    let r = unsafe {
        (&*z).foo()
    };
    assert!(r == 42);

    let mut p = Foo {f: A { f: 42 }};
    let o: *mut Foo<Trait> = &mut p;
    let r = unsafe {
        (&*o).f.foo()
    };
    assert!(r == 42);

    let a: *mut [_] = &mut [1i, 2, 3];
    unsafe {
        let b = (*a)[2];
        assert!(b == 3);
        let len = (*a).len();
        assert!(len == 3);
    }

    let a = &mut [1i, 2, 3] as *mut [_];
    unsafe {
        let b = (*a)[2];
        assert!(b == 3);
        let len = (*a).len();
        assert!(len == 3);
    }

    let c: *mut Foo<[_]> = &mut Foo {f: [1i, 2, 3]};
    unsafe {
        let b = (&*c).f[0];
        assert!(b == 1);
        let len = (&*c).f.len();
        assert!(len == 3);
    }
}