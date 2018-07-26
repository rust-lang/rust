// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// In contrast to `region-escape-via-bound-invariant`, in this case we
// *can* return a value of type `&'x u32`, even though `'x` does not
// appear in the bounds. This is because `&` is contravariant, and so
// we are *actually* returning a `&'y u32`.
//
// See https://github.com/rust-lang/rust/issues/46541 for more details.

// run-pass

#![allow(dead_code)]
#![feature(in_band_lifetimes)]
#![feature(nll)]

trait Trait<'a> { }

impl Trait<'b> for &'a u32 { }

fn foo(x: &'x u32) -> impl Trait<'y>
where 'x: 'y
{
    x
}

fn main() { }
