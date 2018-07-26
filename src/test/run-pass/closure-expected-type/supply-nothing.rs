// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn with_closure<F>(f: F) -> u32
    where F: FnOnce(&u32, &u32) -> u32
{
    f(&22, &44)
}

fn main() {
    let z = with_closure(|x, y| x + y).wrapping_add(1);
    assert_eq!(z, 22 + 44 + 1);
}
