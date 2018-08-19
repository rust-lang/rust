// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn into_inner() -> [u64; 1024] {
    let mut x = 10 + 20;
    [x; 1024]
}

fn main(){
    let x: &[u64] = &[30; 1024];
    assert_eq!(&into_inner()[..], x);
}
