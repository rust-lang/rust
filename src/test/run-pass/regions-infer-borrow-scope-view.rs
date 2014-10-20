// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn view<T>(x: &[T]) -> &[T] {x}

pub fn main() {
    let v = vec!(1i, 2, 3);
    let x = view(v.as_slice());
    let y = view(x.as_slice());
    assert!((v[0] == x[0]) && (v[0] == y[0]));
}
