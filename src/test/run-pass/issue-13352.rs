// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate libc;

use std::thunk::Thunk;

fn foo(_: Thunk) {}

fn main() {
    foo(loop {
        unsafe { libc::exit(0 as libc::c_int); }
    });
    2u + (loop {});
}
