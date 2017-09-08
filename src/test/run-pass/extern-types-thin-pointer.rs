// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that pointers and references to extern types are thin, ie they have the same size and
// alignment as a pointer to ().

#![feature(extern_types)]

use std::mem::{align_of, size_of};

extern {
    type A;
}

fn main() {
    assert_eq!(size_of::<*const A>(), size_of::<*const ()>());
    assert_eq!(align_of::<*const A>(), align_of::<*const ()>());

    assert_eq!(size_of::<*mut A>(), size_of::<*mut ()>());
    assert_eq!(align_of::<*mut A>(), align_of::<*mut ()>());

    assert_eq!(size_of::<&A>(), size_of::<&()>());
    assert_eq!(align_of::<&A>(), align_of::<&()>());

    assert_eq!(size_of::<&mut A>(), size_of::<&mut ()>());
    assert_eq!(align_of::<&mut A>(), align_of::<&mut ()>());
}
