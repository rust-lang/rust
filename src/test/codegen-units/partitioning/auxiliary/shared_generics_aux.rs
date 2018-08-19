// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Zshare-generics=yes

#![crate_type="rlib"]

pub fn generic_fn<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

pub fn use_generic_fn_f32() -> (f32, f32) {
    generic_fn(0.0f32, 1.0f32)
}
