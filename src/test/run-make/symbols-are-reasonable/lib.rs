// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static X: &'static str = "foobarbaz";
pub static Y: &'static [u8] = include_bytes!("lib.rs");

trait Foo { fn dummy(&self) { } }
impl Foo for uint {}

pub fn dummy() {
    // force the vtable to be created
    let _x = &1_usize as &Foo;
}
