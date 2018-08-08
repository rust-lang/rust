// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _foo = &[1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `&[usize; 2]` as `[usize]`

    let _bar = Box::new(1_usize) as std::fmt::Debug;
    //~^ ERROR cast to unsized type: `std::boxed::Box<usize>` as `dyn std::fmt::Debug`

    let _baz = 1_usize as std::fmt::Debug;
    //~^ ERROR cast to unsized type: `usize` as `dyn std::fmt::Debug`

    let _quux = [1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `[usize; 2]` as `[usize]`
}
