// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// compile-flags: -O

use std::collections::HashSet;

#[derive(PartialEq, Debug, Hash, Eq, Clone, PartialOrd, Ord)]
enum MyEnum {
    E0,

    E1,

    E2,
    E3,
    E4,

    E5,
    E6,
    E7,
}


fn main() {
    use MyEnum::*;
    let s: HashSet<_> = [E4, E1].iter().cloned().collect();
    let mut v: Vec<_> = s.into_iter().collect();
    v.sort();

    assert_eq!([E1, E4], &v[..]);
}
