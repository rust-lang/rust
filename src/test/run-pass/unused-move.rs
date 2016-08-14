// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #3878
// Issue Name: Unused move causes a crash
// Abstract: zero-fill to block after drop

// pretty-expanded FIXME #23616

#![allow(path_statements)]
#![allow(unknown_features)]
#![feature(box_syntax)]

pub fn main()
{
    let y: Box<_> = box 1;
    y;
}
