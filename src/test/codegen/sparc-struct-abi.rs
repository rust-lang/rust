// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// Checks that we correctly codegen extern "C" functions returning structs.
// See issue #52638.

// compile-flags: -O --target=sparc64-unknown-linux-gnu --crate-type=rlib
#![feature(no_core, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="freeze"]
trait Freeze { }
#[lang="copy"]
trait Copy { }

#[repr(C)]
pub struct Bool {
    b: bool,
}

// CHECK: define i64 @structbool()
// CHECK-NEXT: start:
// CHECK-NEXT: ret i64 72057594037927936
#[no_mangle]
pub extern "C" fn structbool() -> Bool {
    Bool { b: true }
}
