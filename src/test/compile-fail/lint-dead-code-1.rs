// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unused_variable)];
#[deny(dead_code)];

#[crate_type="lib"];

pub use foo2::Bar2;
mod foo {
    pub struct Bar; //~ ERROR: code is never used
}

mod foo2 {
    pub struct Bar2;
}

pub static pub_static: int = 0;
static priv_static: int = 0; //~ ERROR: code is never used
static used_static: int = 0;
pub static used_static2: int = used_static;
static USED_STATIC: int = 0;

pub type typ = ~UsedStruct4;
pub struct PubStruct();
struct PrivStruct; //~ ERROR: code is never used
struct UsedStruct1 { x: int }
struct UsedStruct2(int);
struct UsedStruct3;
struct UsedStruct4;
// this struct is never used directly, but its method is, so we don't want
// to warn it
struct SemiUsedStruct;
impl SemiUsedStruct {
    fn la_la_la() {}
}

pub enum pub_enum { foo1, bar1 }
enum priv_enum { foo2, bar2 } //~ ERROR: code is never used
enum used_enum { foo3, bar3 }

pub fn pub_fn() {
    used_fn();
    let used_struct1 = UsedStruct1 { x: 1 };
    let used_struct2 = UsedStruct2(1);
    let used_struct3 = UsedStruct3;
    let e = foo3;
    SemiUsedStruct::la_la_la();

    let i = 1;
    match i {
        USED_STATIC => (),
        _ => ()
    }
}
fn priv_fn() { //~ ERROR: code is never used
    let unused_struct = PrivStruct;
}
fn used_fn() {}

fn foo() { //~ ERROR: code is never used
    bar();
    let unused_enum = foo2;
}

fn bar() { //~ ERROR: code is never used
    foo();
}
