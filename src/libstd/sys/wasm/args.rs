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
use mem;
use vec;

pub unsafe fn init(_argc: isize, _argv: *const *const u8) {
    // On wasm these should always be null, so there's nothing for us to do here
}

pub unsafe fn cleanup() {
}

pub fn args() -> Args {
    // When the runtime debugging is enabled we'll link to some extra runtime
    // functions to actually implement this. These are for now just implemented
    // in a node.js script but they're off by default as they're sort of weird
    // in a web-wasm world.
    if !super::DEBUG {
        return Args {
            iter: Vec::new().into_iter(),
            _dont_send_or_sync_me: PhantomData,
        }
    }

    // You'll find the definitions of these in `src/etc/wasm32-shim.js`. These
    // are just meant for debugging and should not be relied on.
    extern {
        fn rust_wasm_args_count() -> usize;
        fn rust_wasm_args_arg_size(a: usize) -> usize;
        fn rust_wasm_args_arg_fill(a: usize, ptr: *mut u8);
    }

    unsafe {
        let cnt = rust_wasm_args_count();
        let mut v = Vec::with_capacity(cnt);
        for i in 0..cnt {
            let n = rust_wasm_args_arg_size(i);
            let mut data = vec![0; n];
            rust_wasm_args_arg_fill(i, data.as_mut_ptr());
            v.push(mem::transmute::<Vec<u8>, OsString>(data));
        }
        Args {
            iter: v.into_iter(),
            _dont_send_or_sync_me: PhantomData,
        }
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
