// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: too big for the current architecture

// FIXME: work properly with higher limits

#[cfg(target_pointer_width = "32")]
fn main() {
    let _big: Option<[u32; (1<<29)-1]> = None;
}

#[cfg(target_pointer_width = "64")]
fn main() {
    let _big: Option<[u32; (1<<45)-1]> = None;
}
