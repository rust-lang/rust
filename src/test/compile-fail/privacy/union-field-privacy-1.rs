// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]
#![feature(untagged_unions)]

mod m {
    pub union U {
        pub a: u8,
        pub(super) b: u8,
        c: u8,
    }
}

fn main() {
    let u = m::U { a: 0 }; // OK
    let u = m::U { b: 0 }; // OK
    let u = m::U { c: 0 }; //~ ERROR field `c` of union `m::U` is private

    let m::U { a } = u; // OK
    let m::U { b } = u; // OK
    let m::U { c } = u; //~ ERROR field `c` of union `m::U` is private
}
