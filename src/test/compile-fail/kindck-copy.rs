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

use std::marker::MarkerTrait;
use std::rc::Rc;

fn assert_copy<T:Copy>() { }

trait Dummy : MarkerTrait { }

#[derive(Copy)]
struct MyStruct {
    x: isize,
    y: isize,
}

struct MyNoncopyStruct {
    x: Box<char>,
}

fn test<'a,T,U:Copy>(_: &'a isize) {
    // lifetime pointers are ok...
    assert_copy::<&'static isize>();
    assert_copy::<&'a isize>();
    assert_copy::<&'a str>();
    assert_copy::<&'a [isize]>();

    // ...unless they are mutable
    assert_copy::<&'static mut isize>(); //~ ERROR `core::marker::Copy` is not implemented
    assert_copy::<&'a mut isize>();  //~ ERROR `core::marker::Copy` is not implemented

    // ~ pointers are not ok
    assert_copy::<Box<isize>>();   //~ ERROR `core::marker::Copy` is not implemented
    assert_copy::<String>();   //~ ERROR `core::marker::Copy` is not implemented
    assert_copy::<Vec<isize> >(); //~ ERROR `core::marker::Copy` is not implemented
    assert_copy::<Box<&'a mut isize>>(); //~ ERROR `core::marker::Copy` is not implemented

    // borrowed object types are generally ok
    assert_copy::<&'a Dummy>();
    assert_copy::<&'a (Dummy+Copy)>();
    assert_copy::<&'static (Dummy+Copy)>();

    // owned object types are not ok
    assert_copy::<Box<Dummy>>(); //~ ERROR `core::marker::Copy` is not implemented
    assert_copy::<Box<Dummy+Copy>>(); //~ ERROR `core::marker::Copy` is not implemented

    // mutable object types are not ok
    assert_copy::<&'a mut (Dummy+Copy)>();  //~ ERROR `core::marker::Copy` is not implemented

    // unsafe ptrs are ok
    assert_copy::<*const isize>();
    assert_copy::<*const &'a mut isize>();

    // regular old ints and such are ok
    assert_copy::<isize>();
    assert_copy::<bool>();
    assert_copy::<()>();

    // tuples are ok
    assert_copy::<(isize,isize)>();

    // structs of POD are ok
    assert_copy::<MyStruct>();

    // structs containing non-POD are not ok
    assert_copy::<MyNoncopyStruct>(); //~ ERROR `core::marker::Copy` is not implemented

    // ref counted types are not ok
    assert_copy::<Rc<isize>>();   //~ ERROR `core::marker::Copy` is not implemented
}

pub fn main() {
}

