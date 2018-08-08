// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Test that changing what a `type` points to does not go unnoticed.

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![allow(unused_variables)]

fn main() { }


#[rustc_if_this_changed]
type TypeAlias = u32;

// The type alias directly affects the type of the field,
// not the enclosing struct:
#[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
struct Struct {
    #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
    x: TypeAlias,
    y: u32
}

#[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
enum Enum {
    Variant1 {
        #[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
        t: TypeAlias
    },
    Variant2(i32)
}

#[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
trait Trait {
    #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
    fn method(&self, _: TypeAlias);
}

struct SomeType;

#[rustc_then_this_would_need(TypeOfItem)] //~ ERROR no path
impl SomeType {
    #[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
    #[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
    fn method(&self, _: TypeAlias) {}
}

#[rustc_then_this_would_need(TypeOfItem)] //~ ERROR OK
type TypeAlias2 = TypeAlias;

#[rustc_then_this_would_need(FnSignature)] //~ ERROR OK
#[rustc_then_this_would_need(TypeckTables)] //~ ERROR OK
fn function(_: TypeAlias) {

}
