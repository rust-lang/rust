// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ffi::OsString;
use fortanix_sgx_abi::ByteBuffer;

pub unsafe fn init(argc: isize, argv: *const *const u8) {
    // See ABI
    let _len: usize = argc as _;
    let _args: *const ByteBuffer = argv as _;

    // TODO
}

pub unsafe fn cleanup() {
}

pub fn args() -> Args {
    Args
}

pub struct Args;

impl Args {
    pub fn inner_debug(&self) -> &[OsString] {
        &[]
    }
}

impl Iterator for Args {
    type Item = OsString;
    fn next(&mut self) -> Option<OsString> {
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl ExactSizeIterator for Args {
    fn len(&self) -> usize {
        0
    }
}

impl DoubleEndedIterator for Args {
    fn next_back(&mut self) -> Option<OsString> {
        None
    }
}
