// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty - token trees can't pretty print

#![feature(macro_rules)]

macro_rules! myfn(
    ( $f:ident, ( $( $x:ident ),* ), $body:block ) => (
        fn $f( $( $x : int),* ) -> int $body
    )
)

myfn!(add, (a,b), { return a+b; } )

pub fn main() {

    macro_rules! mylet(
        ($x:ident, $val:expr) => (
            let $x = $val;
        )
    );

    mylet!(y, 8*2);
    assert_eq!(y, 16);

    myfn!(mult, (a,b), { a*b } );

    assert_eq!(mult(2, add(4,4)), 16);

    macro_rules! actually_an_expr_macro (
        () => ( 16 )
    )

    assert_eq!({ actually_an_expr_macro!() }, 16);

}
