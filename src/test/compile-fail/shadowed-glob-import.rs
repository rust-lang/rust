// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    pub struct Bar;

    pub enum E { V }
    pub use self::E::V;

    pub fn baz() -> bool {}
    pub mod baz { pub fn f() {} }
}

use foo::*;
fn Bar() {}
struct V { x: i32 }
fn baz() {}

fn main() {
    // The function `Bar` shadows the imported struct `Bar` in both namespaces
    Bar();
    let x: Bar = unimplemented!(); //~ ERROR use of undeclared type name `Bar`

    // The struct `V` shadows the imported variant `V` in both namespaces
    let _ = V { x: 0 };
    let _ = V; //~ ERROR `V` is the name of a struct

    // The function `baz` only shadows the imported name `baz` in the value namespace
    let _: () = baz();
    let _ = baz::f();
}
