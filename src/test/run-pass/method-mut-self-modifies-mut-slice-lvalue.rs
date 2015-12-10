// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that an `&mut self` method, when invoked on an lvalue whose
// type is `&mut [u8]`, passes in a pointer to the lvalue and not a
// temporary. Issue #19147.

#![feature(clone_from_slice)]

use std::slice;

trait MyWriter {
    fn my_write(&mut self, buf: &[u8]) -> Result<(), ()>;
}

impl<'a> MyWriter for &'a mut [u8] {
    fn my_write(&mut self, buf: &[u8]) -> Result<(), ()> {
        self.clone_from_slice(buf);

        let write_len = buf.len();
        unsafe {
            *self = slice::from_raw_parts_mut(
                self.as_mut_ptr().offset(write_len as isize),
                self.len() - write_len
            );
        }

        Ok(())
    }
}

fn main() {
    let mut buf = [0; 6];

    {
        let mut writer: &mut [_] = &mut buf;
        writer.my_write(&[0, 1, 2]).unwrap();
        writer.my_write(&[3, 4, 5]).unwrap();
    }

    // If `my_write` is not modifying `buf` in place, then we will
    // wind up with `[3, 4, 5, 0, 0, 0]` because the first call to
    // `my_write()` doesn't update the starting point for the write.

    assert_eq!(buf, [0, 1, 2, 3, 4, 5]);
}
