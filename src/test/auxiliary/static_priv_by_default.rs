// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "lib"]

static private: int = 0;
pub static public: int = 0;

pub struct A(());

impl A {
    fn foo() {}
}

mod foo {
    pub static a: int = 0;
    pub fn b() {}
    pub struct c;
    pub enum d {}

    pub struct A(());

    impl A {
        fn foo() {}
    }

    // these are public so the parent can reexport them.
    pub static reexported_a: int = 0;
    pub fn reexported_b() {}
    pub struct reexported_c;
    pub enum reexported_d {}
}

pub mod bar {
    pub use foo::reexported_a as e;
    pub use foo::reexported_b as f;
    pub use foo::reexported_c as g;
    pub use foo::reexported_d as h;
}

pub static a: int = 0;
pub fn b() {}
pub struct c;
pub enum d {}

static i: int = 0;
fn j() {}
struct k;
enum l {}
