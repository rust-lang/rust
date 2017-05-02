// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #41677. The local variable was winding up with
// a type `Receiver<?T, H>` where `?T` was unconstrained, because we
// failed to enforce the WF obligations and `?T` is a bivariant type
// parameter position.

#![allow(unused_variables, dead_code)]

use std::marker::PhantomData;

trait Handle {
    type Inner;
}

struct ResizingHandle<H>(PhantomData<H>);
impl<H> Handle for ResizingHandle<H> {
    type Inner = H;
}

struct Receiver<T, H: Handle<Inner=T>>(PhantomData<H>);

fn channel<T>(size: usize) -> Receiver<T, ResizingHandle<T>> {
    let rx = Receiver(PhantomData);
    rx
}

fn main() {
}
