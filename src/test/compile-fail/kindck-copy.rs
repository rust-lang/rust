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


use std::rc::Rc;

fn assert_copy<T:Copy>() { }
trait Dummy { }

struct MyStruct {
    x: int,
    y: int,
}

struct MyNoncopyStruct {
    x: Box<int>,
}

fn test<'a,T,U:Copy>(_: &'a int) {
    // lifetime pointers are ok...
    assert_copy::<&'static int>();
    assert_copy::<&'a int>();
    assert_copy::<&'a str>();
    assert_copy::<&'a [int]>();

    // ...unless they are mutable
    assert_copy::<&'static mut int>(); //~ ERROR `core::kinds::Copy` is not implemented
    assert_copy::<&'a mut int>();  //~ ERROR `core::kinds::Copy` is not implemented

    // ~ pointers are not ok
    assert_copy::<Box<int>>();   //~ ERROR `core::kinds::Copy` is not implemented
    assert_copy::<String>();   //~ ERROR `core::kinds::Copy` is not implemented
    assert_copy::<Vec<int> >(); //~ ERROR `core::kinds::Copy` is not implemented
    assert_copy::<Box<&'a mut int>>(); //~ ERROR `core::kinds::Copy` is not implemented

    // borrowed object types are generally ok
    assert_copy::<&'a Dummy>();
    assert_copy::<&'a Dummy+Copy>();
    assert_copy::<&'static Dummy+Copy>();

    // owned object types are not ok
    assert_copy::<Box<Dummy>>(); //~ ERROR `core::kinds::Copy` is not implemented
    assert_copy::<Box<Dummy+Copy>>(); //~ ERROR `core::kinds::Copy` is not implemented

    // mutable object types are not ok
    assert_copy::<&'a mut Dummy+Copy>();  //~ ERROR `core::kinds::Copy` is not implemented

    // closures are like an `&mut` object
    assert_copy::<||>(); //~ ERROR `core::kinds::Copy` is not implemented

    // unsafe ptrs are ok
    assert_copy::<*const int>();
    assert_copy::<*const &'a mut int>();

    // regular old ints and such are ok
    assert_copy::<int>();
    assert_copy::<bool>();
    assert_copy::<()>();

    // tuples are ok
    assert_copy::<(int,int)>();

    // structs of POD are ok
    assert_copy::<MyStruct>();

    // structs containing non-POD are not ok
    assert_copy::<MyNoncopyStruct>(); //~ ERROR `core::kinds::Copy` is not implemented

    // ref counted types are not ok
    assert_copy::<Rc<int>>();   //~ ERROR `core::kinds::Copy` is not implemented
}

pub fn main() {
}

