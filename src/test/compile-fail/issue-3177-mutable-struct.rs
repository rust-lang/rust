// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
// error-pattern: instantiating a type parameter with an incompatible type
struct S<T:Freeze> {
    s: T,
    cant_nest: ()
}

fn main() {
    let a1  = ~S{ s: true, cant_nest: () };
    let _a2 = ~S{ s: a1, cant_nest: () };
}
