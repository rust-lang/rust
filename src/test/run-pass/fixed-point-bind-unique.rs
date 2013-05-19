// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

fn fix_help<A:'static,B:Owned>(f: extern fn(@fn(A) -> B, A) -> B, x: A) -> B {
    return f(|a| fix_help(f, a), x);
}

fn fix<A:'static,B:Owned>(f: extern fn(@fn(A) -> B, A) -> B) -> @fn(A) -> B {
    return |a| fix_help(f, a);
}

fn fact_(f: @fn(v: int) -> int, n: int) -> int {
    // fun fact 0 = 1
    return if n == 0 { 1 } else { n * f(n - 1) };
}

pub fn main() {
    let fact = fix(fact_);
    assert_eq!(fact(5), 120);
    assert_eq!(fact(2), 2);
}
