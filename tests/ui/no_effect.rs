// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![warn(clippy::no_effect)]
#![allow(dead_code)]
#![allow(path_statements)]
#![allow(clippy::deref_addrof)]
#![allow(clippy::redundant_field_names)]
#![feature(untagged_unions)]

struct Unit;
struct Tuple(i32);
struct Struct {
    field: i32,
}
enum Enum {
    Tuple(i32),
    Struct { field: i32 },
}
struct DropUnit;
impl Drop for DropUnit {
    fn drop(&mut self) {}
}
struct DropStruct {
    field: i32,
}
impl Drop for DropStruct {
    fn drop(&mut self) {}
}
struct DropTuple(i32);
impl Drop for DropTuple {
    fn drop(&mut self) {}
}
enum DropEnum {
    Tuple(i32),
    Struct { field: i32 },
}
impl Drop for DropEnum {
    fn drop(&mut self) {}
}
struct FooString {
    s: String,
}
union Union {
    a: u8,
    b: f64,
}

fn get_number() -> i32 {
    0
}
fn get_struct() -> Struct {
    Struct { field: 0 }
}
fn get_drop_struct() -> DropStruct {
    DropStruct { field: 0 }
}

unsafe fn unsafe_fn() -> i32 {
    0
}

struct A(i32);
struct B {
    field: i32,
}
struct C {
    b: B,
}
struct D {
    arr: [i32; 1],
}
const A_CONST: A = A(1);
const B: B = B { field: 1 };
const C: C = C { b: B { field: 1 } };
const D: D = D { arr: [1] };

fn main() {
    let s = get_struct();
    let s2 = get_struct();

    0;
    s2;
    Unit;
    Tuple(0);
    Struct { field: 0 };
    Struct { ..s };
    Union { a: 0 };
    Enum::Tuple(0);
    Enum::Struct { field: 0 };
    5 + 6;
    *&42;
    &6;
    (5, 6, 7);
    box 42;
    ..;
    5..;
    ..5;
    5..6;
    5..=6;
    [42, 55];
    [42, 55][1];
    (42, 55).1;
    [42; 55];
    [42; 55][13];
    let mut x = 0;
    || x += 5;
    let s: String = "foo".into();
    FooString { s: s };
    A_CONST.0 = 2;
    B.field = 2;
    C.b.field = 2;
    D.arr[0] = 2;

    // Do not warn
    get_number();
    unsafe { unsafe_fn() };
    DropUnit;
    DropStruct { field: 0 };
    DropTuple(0);
    DropEnum::Tuple(0);
    DropEnum::Struct { field: 0 };
    let mut a_mut = A(1);
    a_mut.0 = 2;
    let mut b_mut = B { field: 1 };
    b_mut.field = 2;
    let mut c_mut = C { b: B { field: 1 } };
    c_mut.b.field = 2;
    let mut d_mut = D { arr: [1] };
    d_mut.arr[0] = 2;
}
