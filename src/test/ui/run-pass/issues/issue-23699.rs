// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn gimme_a_raw_pointer<T>(_: *const T) { }

fn test<T>(t: T) { }

fn main() {
    // Clearly `pointer` must be of type `*const ()`.
    let pointer = &() as *const _;
    gimme_a_raw_pointer(pointer);

    let t = test as fn (i32);
    t(0i32);
}

