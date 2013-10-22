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
// resolved when we finish typechecking the &fn.


struct Refs { refs: ~[int], n: int }

pub fn main() {
    let e = @mut Refs{refs: ~[], n: 0};
    let _f: &fn() = || error!("{}", e.n);
    e.refs.push(1);
}
