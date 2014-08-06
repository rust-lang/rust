// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Fat<Sized? T> {
    f1: int,
    f2: &'static str,
    ptr: T
}

#[deriving(PartialEq,Eq)]
struct Bar;

#[deriving(PartialEq,Eq)]
struct Bar1 {
    f: int
}

trait ToBar {
    fn to_bar(&self) -> Bar;
    fn to_val(&self) -> int;
}

impl ToBar for Bar {
    fn to_bar(&self) -> Bar {
        *self
    }
    fn to_val(&self) -> int {
        0
    }
}
impl ToBar for Bar1 {
    fn to_bar(&self) -> Bar {
        Bar
    }
    fn to_val(&self) -> int {
        self.f
    }
}

// x is a fat pointer
fn foo(x: &Fat<ToBar>) {
    assert!(x.f1 == 5);
    assert!(x.f2 == "some str");
    assert!(x.ptr.to_bar() == Bar);
    assert!(x.ptr.to_val() == 42);

    let y = &x.ptr;
    assert!(y.to_bar() == Bar);
    assert!(y.to_val() == 42);
}

fn bar(x: &ToBar) {
    assert!(x.to_bar() == Bar);
    assert!(x.to_val() == 42);
}

fn baz(x: &Fat<Fat<ToBar>>) {
    assert!(x.f1 == 5);
    assert!(x.f2 == "some str");
    assert!(x.ptr.f1 == 8);
    assert!(x.ptr.f2 == "deep str");
    assert!(x.ptr.ptr.to_bar() == Bar);
    assert!(x.ptr.ptr.to_val() == 42);

    let y = &x.ptr.ptr;
    assert!(y.to_bar() == Bar);
    assert!(y.to_val() == 42);

}

pub fn main() {
    let f1 = Fat { f1: 5, f2: "some str", ptr: Bar1 {f :42} };
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<ToBar> = f2;
    foo(f3);
    let f4: &Fat<ToBar> = &f1;
    foo(f4);
    let f5: &Fat<ToBar> = &Fat { f1: 5, f2: "some str", ptr: Bar1 {f :42} };
    foo(f5);

    // Zero size object.
    let f6: &Fat<ToBar> = &Fat { f1: 5, f2: "some str", ptr: Bar };
    assert!(f6.ptr.to_bar() == Bar);

    // &*
    let f7: Box<ToBar> = box Bar1 {f :42};
    bar(&*f7);

    // Deep nesting
    let f1 =
        Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: Bar1 {f :42}} };
    baz(&f1);
    let f2 = &f1;
    baz(f2);
    let f3: &Fat<Fat<ToBar>> = f2;
    baz(f3);
    let f4: &Fat<Fat<ToBar>> = &f1;
    baz(f4);
    let f5: &Fat<Fat<ToBar>> =
        &Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: Bar1 {f :42}} };
    baz(f5);
}
