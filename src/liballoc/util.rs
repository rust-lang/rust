// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc(hidden)]

use core::mem;
use core::raw;

#[inline]
#[deprecated]
pub fn get_box_size(body_size: uint, body_align: uint) -> uint {
    let header_size = mem::size_of::<raw::Box<()>>();
    let total_size = align_to(header_size, body_align) + body_size;
    total_size
}

// Rounds size to the next alignment. Alignment is required to be a power of
// two.
#[inline]
fn align_to(size: uint, align: uint) -> uint {
    assert!(align != 0);
    (size + align - 1) & !(align - 1)
}

// FIXME(#14344): When linking liballoc with libstd, this library will be linked
//                as an rlib (it only exists as an rlib). It turns out that an
//                optimized standard library doesn't actually use *any* symbols
//                from this library. Everything is inlined and optimized away.
//                This means that linkers will actually omit the object for this
//                file, even though it may be needed in the future.
//
//                To get around this for now, we define a dummy symbol which
//                will never get inlined so the stdlib can call it. The stdlib's
//                reference to this symbol will cause this library's object file
//                to get linked in to libstd successfully (the linker won't
//                optimize it out).
#[deprecated]
#[doc(hidden)]
pub fn make_stdlib_link_work() {}

