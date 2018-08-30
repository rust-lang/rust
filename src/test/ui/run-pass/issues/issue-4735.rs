// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::mem::transmute;

struct NonCopyable(*const u8);

impl Drop for NonCopyable {
    fn drop(&mut self) {
        let NonCopyable(p) = *self;
        let _v = unsafe { transmute::<*const u8, Box<isize>>(p) };
    }
}

pub fn main() {
    let t = Box::new(0);
    let p = unsafe { transmute::<Box<isize>, *const u8>(t) };
    let _z = NonCopyable(p);
}
