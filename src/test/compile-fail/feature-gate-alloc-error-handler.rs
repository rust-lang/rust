// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::alloc::Layout;

#[alloc_error_handler] //~ ERROR #[alloc_error_handler] is an unstable feature (see issue #51540)
fn oom(info: Layout) -> ! {
    loop {}
}
