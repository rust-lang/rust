// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//compile-pass

#![feature(const_fn_union)]

fn main() {}

static FOO: u32 = 42;

union Foo {
    f: Float,
    r: &'static u32,
}

#[cfg(target_pointer_width="64")]
type Float = f64;

#[cfg(target_pointer_width="32")]
type Float = f32;

static BAR: Float = unsafe { Foo { r: &FOO }.f };
