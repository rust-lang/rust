// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::cast::transmute;
use std::libc::c_void;

struct NonCopyable(*c_void);

impl Drop for NonCopyable {
    fn drop(&mut self) {
        let p = **self;
        let _v = unsafe { transmute::<*c_void, ~int>(p) };
    }
}

fn main() {
    let t = ~0;
    let p = unsafe { transmute::<~int, *c_void>(t) };
    let _z = NonCopyable(p);
}
