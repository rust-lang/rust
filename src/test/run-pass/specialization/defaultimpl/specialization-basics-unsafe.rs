// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Tests a variety of basic specialization scenarios and method
// dispatch for them.

unsafe trait Foo {
    fn foo(&self) -> &'static str;
}

unsafe default impl<T> Foo for T {
    fn foo(&self) -> &'static str {
        "generic"
    }
}

unsafe default impl<T: Clone> Foo for T {
    fn foo(&self) -> &'static str {
        "generic Clone"
    }
}

unsafe default impl<T, U> Foo for (T, U) where T: Clone, U: Clone {
    fn foo(&self) -> &'static str {
        "generic pair"
    }
}

unsafe default impl<T: Clone> Foo for (T, T) {
    fn foo(&self) -> &'static str {
        "generic uniform pair"
    }
}

unsafe default impl Foo for (u8, u32) {
    fn foo(&self) -> &'static str {
        "(u8, u32)"
    }
}

unsafe default impl Foo for (u8, u8) {
    fn foo(&self) -> &'static str {
        "(u8, u8)"
    }
}

unsafe default impl<T: Clone> Foo for Vec<T> {
    fn foo(&self) -> &'static str {
        "generic Vec"
    }
}

unsafe impl Foo for Vec<i32> {
    fn foo(&self) -> &'static str {
        "Vec<i32>"
    }
}

unsafe impl Foo for String {
    fn foo(&self) -> &'static str {
        "String"
    }
}

unsafe impl Foo for i32 {
    fn foo(&self) -> &'static str {
        "i32"
    }
}

struct NotClone;

unsafe trait MyMarker {}
unsafe default impl<T: Clone + MyMarker> Foo for T {
    fn foo(&self) -> &'static str {
        "generic Clone + MyMarker"
    }
}

#[derive(Clone)]
struct MarkedAndClone;
unsafe impl MyMarker for MarkedAndClone {}

fn  main() {
    assert!(NotClone.foo() == "generic");
    assert!(0u8.foo() == "generic Clone");
    assert!(vec![NotClone].foo() == "generic");
    assert!(vec![0u8].foo() == "generic Vec");
    assert!(vec![0i32].foo() == "Vec<i32>");
    assert!(0i32.foo() == "i32");
    assert!(String::new().foo() == "String");
    assert!(((), 0).foo() == "generic pair");
    assert!(((), ()).foo() == "generic uniform pair");
    assert!((0u8, 0u32).foo() == "(u8, u32)");
    assert!((0u8, 0u8).foo() == "(u8, u8)");
    assert!(MarkedAndClone.foo() == "generic Clone + MyMarker");
}
