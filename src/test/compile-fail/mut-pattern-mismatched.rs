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
    let foo = &mut 1is;

    // (separate lines to ensure the spans are accurate)

     let &_ //~ ERROR expected `&mut isize`, found `&_`
        = foo;
    let &mut _ = foo;

    let bar = &1is;
    let &_ = bar;
    let &mut _ //~ ERROR expected `&isize`, found `&mut _`
         = bar;
}
