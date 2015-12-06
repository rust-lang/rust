// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![no_std]
#![allow(unused_variables)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![deny(dead_code)]

#![crate_type="lib"]

pub use foo2::Bar2;

mod foo {
    pub struct Bar; //~ ERROR: struct is never used
}

mod foo2 {
    pub struct Bar2;
}

pub static pub_static: isize = 0;
static priv_static: isize = 0; //~ ERROR: static item is never used
const used_static: isize = 0;
pub static used_static2: isize = used_static;
const USED_STATIC: isize = 0;
const STATIC_USED_IN_ENUM_DISCRIMINANT: isize = 10;

pub const pub_const: isize = 0;
const priv_const: isize = 0; //~ ERROR: constant item is never used
const used_const: isize = 0;
pub const used_const2: isize = used_const;
const USED_CONST: isize = 1;
const CONST_USED_IN_ENUM_DISCRIMINANT: isize = 11;

pub type typ = *const UsedStruct4;
pub struct PubStruct;
struct PrivStruct; //~ ERROR: struct is never used
struct UsedStruct1 {
    #[allow(dead_code)]
    x: isize
}
struct UsedStruct2(isize);
struct UsedStruct3;
struct UsedStruct4;
// this struct is never used directly, but its method is, so we don't want
// to warn it
struct SemiUsedStruct;
impl SemiUsedStruct {
    fn la_la_la() {}
}
struct StructUsedAsField;
pub struct StructUsedInEnum;
struct StructUsedInGeneric;
pub struct PubStruct2 {
    #[allow(dead_code)]
    struct_used_as_field: *const StructUsedAsField
}

pub enum pub_enum { foo1, bar1 }
pub enum pub_enum2 { a(*const StructUsedInEnum) }
pub enum pub_enum3 {
    Foo = STATIC_USED_IN_ENUM_DISCRIMINANT,
    Bar = CONST_USED_IN_ENUM_DISCRIMINANT,
}

enum priv_enum { foo2, bar2 } //~ ERROR: enum is never used
enum used_enum {
    foo3,
    bar3 //~ ERROR variant is never used
}

fn f<T>() {}

pub fn pub_fn() {
    used_fn();
    let used_struct1 = UsedStruct1 { x: 1 };
    let used_struct2 = UsedStruct2(1);
    let used_struct3 = UsedStruct3;
    let e = used_enum::foo3;
    SemiUsedStruct::la_la_la();

    let i = 1;
    match i {
        USED_STATIC => (),
        USED_CONST => (),
        _ => ()
    }
    f::<StructUsedInGeneric>();
}
fn priv_fn() { //~ ERROR: function is never used
    let unused_struct = PrivStruct;
}
fn used_fn() {}

fn foo() { //~ ERROR: function is never used
    bar();
    let unused_enum = priv_enum::foo2;
}

fn bar() { //~ ERROR: function is never used
    foo();
}

// Code with #[allow(dead_code)] should be marked live (and thus anything it
// calls is marked live)
#[allow(dead_code)]
fn g() { h(); }
fn h() {}
