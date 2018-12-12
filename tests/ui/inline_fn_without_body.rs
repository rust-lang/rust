// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::inline_fn_without_body)]
#![allow(clippy::inline_always)]

trait Foo {
    #[inline]
    fn default_inline();

    #[inline(always)]
    fn always_inline();

    #[inline(never)]
    fn never_inline();

    #[inline]
    fn has_body() {}
}

fn main() {}
