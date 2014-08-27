// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(if_let)]

pub fn main() {
    let x = Some(3i);
    if let Some(y) = x {
        assert_eq!(y, 3i);
    } else {
        fail!("if-let failed");
    }
    let mut worked = false;
    if let Some(_) = x {
        worked = true;
    }
    assert!(worked);
    let clause: uint;
    if let None = Some("test") {
        clause = 1;
    } else if 4u > 5 {
        clause = 2;
    } else if let Ok(()) = Err::<(),&'static str>("test") {
        clause = 3;
    } else {
        clause = 4;
    }
    assert_eq!(clause, 4u);

    if 3i > 4 {
        fail!("bad math");
    } else if let 1 = 2i {
        fail!("bad pattern match");
    }

    enum Foo {
        One,
        Two(uint),
        Three(String, int)
    }

    let foo = Three("three".to_string(), 42i);
    if let One = foo {
        fail!("bad pattern match");
    } else if let Two(_x) = foo {
        fail!("bad pattern match");
    } else if let Three(s, _) = foo {
        assert_eq!(s.as_slice(), "three");
    } else {
        fail!("bad else");
    }

    if false {
        fail!("wat");
    } else if let a@Two(_) = Two(42u) {
        if let Two(b) = a {
            assert_eq!(b, 42u);
        } else {
            fail!("fail in nested if-let");
        }
    }
}
