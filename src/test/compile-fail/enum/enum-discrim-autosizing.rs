// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// With no repr attribute the discriminant will default to isize.
// On 32-bit architectures this is equivalent to i32 so the variants
// collide. On other architectures we need compilation to fail anyway,
// so force the repr.
#[cfg_attr(not(target_pointer_width = "32"), repr(i32))]
enum Eu64 {
    Au64 = 0,
    Bu64 = 0x8000_0000_0000_0000 //~ERROR already exists
}

