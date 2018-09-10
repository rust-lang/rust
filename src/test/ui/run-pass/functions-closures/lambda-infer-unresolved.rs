// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This should typecheck even though the type of e is not fully
// resolved when we finish typechecking the ||.


struct Refs { refs: Vec<isize> , n: isize }

pub fn main() {
    let mut e = Refs{refs: vec![], n: 0};
    let _f = || println!("{}", e.n);
    let x: &[isize] = &e.refs;
    assert_eq!(x.len(), 0);
}
