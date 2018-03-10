// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ffi::OsString;
use marker::PhantomData;
use vec;
use sys::ArgsSysCall;

pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
    // On wasm these should always be null, so there's nothing for us to do here
}

pub unsafe fn cleanup() {
}

pub fn args() -> Args {
    let v = ArgsSysCall::perform();
    Args {
        iter: v.into_iter(),
        _dont_send_or_sync_me: PhantomData,
    }
}

pub struct Args {
    iter: vec::IntoIter<OsString>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl Args {
    pub fn inner_debug(&self) -> &[OsString] {
        self.iter.as_slice()
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        self.iter.next_back()
    }
}
