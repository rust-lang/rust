// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pure fn pure_borrow(_x: &int, _y: ()) {}

fn test1(x: @mut ~int) {
    // Here, evaluating the second argument actually invalidates the
    // first borrow, even though it occurs outside of the scope of the
    // borrow!
    pure_borrow(*x, *x = ~5);  //~ ERROR illegal borrow unless pure
    //~^ NOTE impure due to assigning to dereference of mutable @ pointer
}

fn test2() {
    let mut x = ~1;

    // Same, but for loanable data:

    pure_borrow(x, x = ~5);  //~ ERROR assigning to mutable local variable prohibited due to outstanding loan
    //~^ NOTE loan of mutable local variable granted here

    copy x;
}

fn main() {
}