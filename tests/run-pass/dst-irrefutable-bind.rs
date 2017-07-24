// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Test<T: ?Sized>(T);

fn main() {
    let x = Test([1,2,3]);
    let x : &Test<[i32]> = &x;

    let & ref _y = x;

    // Make sure binding to a fat pointer behind a reference
    // still works
    let slice = &[1,2,3];
    let x = Test(&slice);
    let Test(&_slice) = x;
}
