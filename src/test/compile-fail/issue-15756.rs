// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::Chunks;
use std::slice::ChunksMut;

fn dft_iter<'a, T>(arg1: Chunks<'a,T>, arg2: ChunksMut<'a,T>)
{
    for
    &mut something
//~^ ERROR the trait `core::marker::Sized` is not implemented for the type `[T]`
    in arg2
    {
    }
}

fn main() {}
