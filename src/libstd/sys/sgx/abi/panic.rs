// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io::{self, Write};
use slice::from_raw_parts_mut;

extern "C" {
    fn take_debug_panic_buf_ptr() -> *mut u8;
    static DEBUG: u8;
}

pub(crate) struct SgxPanicOutput(Option<&'static mut [u8]>);

impl SgxPanicOutput {
    pub(crate) fn new() -> Option<Self> {
        if unsafe { DEBUG == 0 } {
            None
        } else {
            Some(SgxPanicOutput(None))
        }
    }

    fn init(&mut self) -> &mut &'static mut [u8] {
        self.0.get_or_insert_with(|| unsafe {
            let ptr = take_debug_panic_buf_ptr();
            if ptr.is_null() {
                &mut []
            } else {
                from_raw_parts_mut(ptr, 1024)
            }
        })
    }
}

impl Write for SgxPanicOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.init().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.init().flush()
    }
}

#[no_mangle]
pub extern "C" fn panic_msg(msg: &str) -> ! {
    let _ = SgxPanicOutput::new().map(|mut out| out.write(msg.as_bytes()));
    unsafe { panic_exit(); }
}

extern "C" { pub fn panic_exit() -> !; }
