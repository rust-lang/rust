// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Try to initialise a DST struct where the lost information is deeply nested.
// This is an error because it requires an unsized rvalue. This is a problem
// because it would require stack allocation of an unsized temporary (*g in the
// test).

struct Fat<Sized? T> {
    ptr: T
}

pub fn main() {
    let f: Fat<[int, ..3]> = Fat { ptr: [5i, 6, 7] };
    let g: &Fat<[int]> = &f;
    let h: &Fat<Fat<[int]>> = &Fat { ptr: *g };
    //~^ ERROR the trait `core::kinds::Sized` is not implemented
}
