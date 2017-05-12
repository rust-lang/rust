// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #41936. The coerce-unsized trait check in
// coherence was using subtyping, which triggered variance
// computation, which failed because it required type info for fields
// that had not (yet) been computed.

#![feature(unsize)]
#![feature(coerce_unsized)]

use std::{marker,ops};

// Change the array to a non-array, and error disappears
// Adding a new field to the end keeps the error
struct LogDataBuf([u8;8]);

struct Aref<T: ?Sized>
{
    // Inner structure triggers the error, removing the inner removes the message.
    ptr: Box<ArefInner<T>>,
}
impl<T: ?Sized + marker::Unsize<U>, U: ?Sized> ops::CoerceUnsized<Aref<U>> for Aref<T> {}

struct ArefInner<T: ?Sized>
{
    // Even with this field commented out, the error is raised.
    data: T,
}

fn main(){}
