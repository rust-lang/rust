// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]

enum Helper<T, U> {
    T(T, [!; 0]),
    #[allow(dead_code)]
    U(U),
}

fn transmute<T, U>(t: T) -> U {
    let Helper::U(u) = Helper::T(t, []);
    //~^ ERROR refutable pattern in local binding: `T(_, _)` not covered
    u
}

fn main() {
    println!("{:?}", transmute::<&str, (*const u8, u64)>("type safety"));
}
