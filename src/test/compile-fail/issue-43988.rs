// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(stmt_expr_attributes)]

fn main() {

    #[inline]
    let _a = 4;
    //~^^ ERROR attribute should be applied to function or closure


    #[inline(XYZ)]
    let _b = 4;
    //~^^ ERROR attribute should be applied to function or closure

    #[repr(nothing)]
    let _x = 0;
    //~^^ ERROR attribute should not be applied to a statement

    #[repr(something_not_real)]
    loop {
        ()
    };
    //~^^^^ ERROR attribute should not be applied to an expression

    #[repr]
    let _y = "123";
    //~^^ ERROR attribute should not be applied to a statement
    //~| WARN `repr` attribute must have a hint


    fn foo() {}

    #[inline(ABC)]
    foo();
    //~^^ ERROR attribute should be applied to function or closure

    let _z = #[repr] 1;
    //~^ ERROR attribute should not be applied to an expression
    //~| WARN `repr` attribute must have a hint
}
