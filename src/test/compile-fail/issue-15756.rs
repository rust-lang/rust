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
use std::slice::MutChunks;

fn dft_iter<'a, T>(arg1: Chunks<'a,T>, arg2: MutChunks<'a,T>)
{
    for
    &something
//~^ ERROR the trait `core::kinds::Sized` is not implemented for the type `[T]`
    in arg2
    {
    }
}

fn main() {}
