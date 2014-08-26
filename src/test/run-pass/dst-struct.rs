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

// x is a fat pointer
fn foo(x: &Fat<[int]>) {
    let y = &x.ptr;
    assert!(x.ptr.len() == 3);
    assert!(y[0] == 1);
    assert!(x.ptr[1] == 2);
    assert!(x.f1 == 5);
    assert!(x.f2 == "some str");
}

fn foo2<T:ToBar>(x: &Fat<[T]>) {
    let y = &x.ptr;
    let bar = Bar;
    assert!(x.ptr.len() == 3);
    assert!(y[0].to_bar() == bar);
    assert!(x.ptr[1].to_bar() == bar);
    assert!(x.f1 == 5);
    assert!(x.f2 == "some str");
}

fn foo3(x: &Fat<Fat<[int]>>) {
    let y = &x.ptr.ptr;
    assert!(x.f1 == 5);
    assert!(x.f2 == "some str");
    assert!(x.ptr.f1 == 8);
    assert!(x.ptr.f2 == "deep str");
    assert!(x.ptr.ptr.len() == 3);
    assert!(y[0] == 1);
    assert!(x.ptr.ptr[1] == 2);
}


#[deriving(PartialEq,Eq)]
struct Bar;

trait ToBar {
    fn to_bar(&self) -> Bar;
}

impl ToBar for Bar {
    fn to_bar(&self) -> Bar {
        *self
    }
}

pub fn main() {
    // With a vec of ints.
    let f1 = Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&f1);
    let f2 = &f1;
    foo(f2);
    let f3: &Fat<[int]> = f2;
    foo(f3);
    let f4: &Fat<[int]> = &f1;
    foo(f4);
    let f5: &Fat<[int]> = &Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(f5);

    // With a vec of Bars.
    let bar = Bar;
    let f1 = Fat { f1: 5, f2: "some str", ptr: [bar, bar, bar] };
    foo2(&f1);
    let f2 = &f1;
    foo2(f2);
    let f3: &Fat<[Bar]> = f2;
    foo2(f3);
    let f4: &Fat<[Bar]> = &f1;
    foo2(f4);
    let f5: &Fat<[Bar]> = &Fat { f1: 5, f2: "some str", ptr: [bar, bar, bar] };
    foo2(f5);

    // Assignment.
    let f5: &mut Fat<[int]> = &mut Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    f5.ptr[1] = 34;
    assert!(f5.ptr[0] == 1);
    assert!(f5.ptr[1] == 34);
    assert!(f5.ptr[2] == 3);

    // Zero size vec.
    let f5: &Fat<[int]> = &Fat { f1: 5, f2: "some str", ptr: [] };
    assert!(f5.ptr.len() == 0);
    let f5: &Fat<[Bar]> = &Fat { f1: 5, f2: "some str", ptr: [] };
    assert!(f5.ptr.len() == 0);

    // Deeply nested.
    let f1 = Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: [1, 2, 3]} };
    foo3(&f1);
    let f2 = &f1;
    foo3(f2);
    let f3: &Fat<Fat<[int]>> = f2;
    foo3(f3);
    let f4: &Fat<Fat<[int]>> = &f1;
    foo3(f4);
    let f5: &Fat<Fat<[int]>> =
        &Fat { f1: 5, f2: "some str", ptr: Fat { f1: 8, f2: "deep str", ptr: [1, 2, 3]} };
    foo3(f5);

    // Box.
    let f1 = box [1i, 2, 3];
    assert!((*f1)[1] == 2);
    let f2: Box<[int]> = f1;
    assert!((*f2)[1] == 2);

    // Nested Box.
    let f1 : Box<Fat<[int, ..3]>> = box Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&*f1);
    let f2 : Box<Fat<[int]>> = f1;
    foo(&*f2);
    let f3 : Box<Fat<[int]>> = box Fat { f1: 5, f2: "some str", ptr: [1, 2, 3] };
    foo(&*f3);
}
