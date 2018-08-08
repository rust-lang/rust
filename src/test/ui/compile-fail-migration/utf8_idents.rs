// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

fn foo<
    'β, //~ ERROR non-ascii idents are not fully supported
    γ  //~ ERROR non-ascii idents are not fully supported
>() {}

struct X {
    δ: usize //~ ERROR non-ascii idents are not fully supported
}

pub fn main() {
    let α = 0.00001f64; //~ ERROR non-ascii idents are not fully supported
}
