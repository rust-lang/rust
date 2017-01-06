// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let Box(a) = loop { };
    //~^ ERROR field `0` of struct `std::boxed::Box` is private

    // (The below is a trick to allow compiler to infer a type for
    // variable `a` without attempting to ascribe a type to the
    // pattern or otherwise attempting to name the Box type, which
    // would run afoul of issue #22207)
    let _b: *mut i32 = *a;
}
