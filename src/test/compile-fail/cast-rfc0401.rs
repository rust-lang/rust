// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn illegal_cast<U:?Sized,V:?Sized>(u: *const U) -> *const V
{
    u as *const V
    //~^ ERROR casting
    //~^^ NOTE vtable kinds
}

fn illegal_cast_2<U:?Sized>(u: *const U) -> *const str
{
    u as *const str
    //~^ ERROR casting
    //~^^ NOTE vtable kinds
}

trait Foo { fn foo(&self) {} }
impl<T> Foo for T {}

trait Bar { fn foo(&self) {} }
impl<T> Bar for T {}

enum E {
    A, B
}

fn main()
{
    let f: f32 = 1.2;
    let v = 0 as *const u8;
    let fat_v : *const [u8] = unsafe { &*(0 as *const [u8; 1])};
    let fat_sv : *const [i8] = unsafe { &*(0 as *const [i8; 1])};
    let foo: &Foo = &f;

    let _ = v as &u8; //~ ERROR non-scalar
    let _ = v as E; //~ ERROR non-scalar
    let _ = v as fn(); //~ ERROR non-scalar
    let _ = v as (u32,); //~ ERROR non-scalar
    let _ = Some(&v) as *const u8; //~ ERROR non-scalar

    let _ = v as f32;
    //~^ ERROR casting
    //~^^ HELP through a usize first
    let _ = main as f64;
    //~^ ERROR casting
    //~^^ HELP through a usize first
    let _ = &v as usize;
    //~^ ERROR casting
    //~^^ HELP through a raw pointer first
    let _ = f as *const u8;
    //~^ ERROR casting
    //~^^ HELP through a usize first
    let _ = 3 as bool;
    //~^ ERROR cannot cast as `bool`
    //~^^ HELP compare with zero
    let _ = E::A as bool;
    //~^ ERROR cannot cast as `bool`
    //~^^ HELP compare with zero
    let _ = 0x61u32 as char; //~ ERROR only `u8` can be cast

    let _ = false as f32;
    //~^ ERROR casting
    //~^^ HELP through an integer first
    let _ = E::A as f32;
    //~^ ERROR casting
    //~^^ HELP through an integer first
    let _ = 'a' as f32;
    //~^ ERROR casting
    //~^^ HELP through an integer first

    let _ = false as *const u8;
    //~^ ERROR casting
    //~^^ HELP through a usize first
    let _ = E::A as *const u8;
    //~^ ERROR casting
    //~^^ HELP through a usize first
    let _ = 'a' as *const u8;
    //~^ ERROR casting
    //~^^ HELP through a usize first

    let _ = 42usize as *const [u8]; //~ ERROR casting
    let _ = v as *const [u8]; //~ ERROR casting
    let _ = fat_v as *const Foo;
    //~^ ERROR `core::marker::Sized` is not implemented for the type `[u8]`
    let _ = foo as *const str; //~ ERROR casting
    let _ = foo as *mut str; //~ ERROR casting
    let _ = main as *mut str; //~ ERROR casting
    let _ = &f as *mut f32; //~ ERROR casting
    let _ = &f as *const f64; //~ ERROR casting
    let _ = fat_sv as usize;
    //~^ ERROR casting
    //~^^ HELP through a thin pointer first

    let a : *const str = "hello";
    let _ = a as *const Foo;
    //~^ ERROR `core::marker::Sized` is not implemented for the type `str`

    // check no error cascade
    let _ = main.f as *const u32; //~ ERROR attempted access of field

    let cf: *const Foo = &0;
    let _ = cf as *const [u16];
    //~^ ERROR casting
    //~^^ NOTE vtable kinds
    let _ = cf as *const Bar;
    //~^ ERROR casting
    //~^^ NOTE vtable kinds
}
