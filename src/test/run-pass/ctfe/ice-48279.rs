// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// https://github.com/rust-lang/rust/issues/48279

#![feature(const_fn)]

#[derive(PartialEq, Eq)]
pub struct NonZeroU32 {
    value: u32
}

impl NonZeroU32 {
    const unsafe fn new_unchecked(value: u32) -> Self {
        NonZeroU32 { value }
    }
}

//pub const FOO_ATOM: NonZeroU32 = unsafe { NonZeroU32::new_unchecked(7) };
pub const FOO_ATOM: NonZeroU32 = unsafe { NonZeroU32 { value: 7 } };

fn main() {
    match None {
        Some(FOO_ATOM) => {}
        _ => {}
    }
}
