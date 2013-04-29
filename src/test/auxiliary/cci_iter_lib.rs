// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[link(name="cci_iter_lib", vers="0.0")];

#[inline]
pub fn iter<T>(v: &[T], f: &fn(&T)) {
    let mut i = 0u;
    let n = vec::len(v);
    while i < n {
        f(&v[i]);
        i += 1u;
    }
}
