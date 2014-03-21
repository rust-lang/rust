// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test which of the builtin types are considered POD.

#[feature(managed_boxes)];

use std::rc::Rc;

fn assert_pod<T:Pod>() { }
trait Dummy { }

struct MyStruct {
    x: int,
    y: int,
}

struct MyNonpodStruct {
    x: ~int,
}

fn test<'a,T,U:Pod>(_: &'a int) {
    // lifetime pointers are ok...
    assert_pod::<&'static int>();
    assert_pod::<&'a int>();
    assert_pod::<&'a str>();
    assert_pod::<&'a [int]>();

    // ...unless they are mutable
    assert_pod::<&'static mut int>(); //~ ERROR does not fulfill `Pod`
    assert_pod::<&'a mut int>();  //~ ERROR does not fulfill `Pod`

    // ~ pointers are not ok
    assert_pod::<~int>();   //~ ERROR does not fulfill `Pod`
    assert_pod::<~str>();   //~ ERROR does not fulfill `Pod`
    assert_pod::<Vec<int> >(); //~ ERROR does not fulfill `Pod`
    assert_pod::<~&'a mut int>(); //~ ERROR does not fulfill `Pod`

    // borrowed object types are generally ok
    assert_pod::<&'a Dummy>();
    assert_pod::<&'a Dummy:Pod>();
    assert_pod::<&'static Dummy:Pod>();

    // owned object types are not ok
    assert_pod::<~Dummy>(); //~ ERROR does not fulfill `Pod`
    assert_pod::<~Dummy:Pod>(); //~ ERROR does not fulfill `Pod`

    // mutable object types are not ok
    assert_pod::<&'a mut Dummy:Pod>();  //~ ERROR does not fulfill `Pod`

    // closures are like an `&mut` object
    assert_pod::<||>(); //~ ERROR does not fulfill `Pod`

    // unsafe ptrs are ok
    assert_pod::<*int>();
    assert_pod::<*&'a mut int>();

    // regular old ints and such are ok
    assert_pod::<int>();
    assert_pod::<bool>();
    assert_pod::<()>();

    // tuples are ok
    assert_pod::<(int,int)>();

    // structs of POD are ok
    assert_pod::<MyStruct>();

    // structs containing non-POD are not ok
    assert_pod::<MyNonpodStruct>(); //~ ERROR does not fulfill `Pod`

    // managed or ref counted types are not ok
    assert_pod::<@int>();   //~ ERROR does not fulfill `Pod`
    assert_pod::<Rc<int>>();   //~ ERROR does not fulfill `Pod`
}

pub fn main() {
}

