// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn works<T>(x: T) -> Vec<T> { vec![x] }

fn also_works<T: Clone>(x: T) -> Vec<T> { vec![x] }

fn main() {
    let _: Vec<usize> = works(0);
    let _: Vec<usize> = also_works(0);
    let _ = works(0);
    let _ = also_works(0);
}
