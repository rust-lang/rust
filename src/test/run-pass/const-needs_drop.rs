// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

struct Trivial(u8, f32);

struct NonTrivial(u8, String);

const CONST_U8: bool = mem::needs_drop::<u8>();
const CONST_STRING: bool = mem::needs_drop::<String>();
const CONST_TRIVIAL: bool = mem::needs_drop::<Trivial>();
const CONST_NON_TRIVIAL: bool = mem::needs_drop::<NonTrivial>();

static STATIC_U8: bool = mem::needs_drop::<u8>();
static STATIC_STRING: bool = mem::needs_drop::<String>();
static STATIC_TRIVIAL: bool = mem::needs_drop::<Trivial>();
static STATIC_NON_TRIVIAL: bool = mem::needs_drop::<NonTrivial>();

fn main() {
    assert!(!CONST_U8);
    assert!(CONST_STRING);
    assert!(!CONST_TRIVIAL);
    assert!(CONST_NON_TRIVIAL);

    assert!(!STATIC_U8);
    assert!(STATIC_STRING);
    assert!(!STATIC_TRIVIAL);
    assert!(STATIC_NON_TRIVIAL);
}
