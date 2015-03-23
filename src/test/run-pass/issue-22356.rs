// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::marker::{PhantomData, PhantomFn};

pub struct Handle<T, I>(T, I);

impl<T, I> Handle<T, I> {
    pub fn get_info(&self) -> &I {
        let Handle(_, ref info) = *self;
        info
    }
}

pub struct BufferHandle<D: Device, T> {
    raw: RawBufferHandle<D>,
    _marker: PhantomData<T>,
}

impl<D: Device, T> BufferHandle<D, T> {
    pub fn get_info(&self) -> &String {
        self.raw.get_info()
    }
}

pub type RawBufferHandle<D: Device> = Handle<<D as Device>::Buffer, String>;

pub trait Device: PhantomFn<Self> {
    type Buffer;
}

fn main() {}
