// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_snake_case)]

register_long_diagnostics! {

E0668: r##"
Malformed inline assembly rejected by LLVM.

LLVM checks the validity of the constraints and the assembly string passed to
it. This error implies that LLVM seems something wrong with the inline
assembly call.

In particular, it can happen if you forgot the closing bracket of a register
constraint (see issue #51430):
```ignore (error-emitted-at-codegen-which-cannot-be-handled-by-compile_fail)
#![feature(asm)]

fn main() {
    let rax: u64;
    unsafe {
        asm!("" :"={rax"(rax));
        println!("Accumulator is: {}", rax);
    }
}
```
"##,

E0669: r##"
Cannot convert inline assembly operand to a single LLVM value.

This error usually happens when trying to pass in a value to an input inline
assembly operand that is actually a pair of values. In particular, this can
happen when trying to pass in a slice, for instance a `&str`. In Rust, these
values are represented internally as a pair of values, the pointer and its
length. When passed as an input operand, this pair of values can not be
coerced into a register and thus we must fail with an error.
"##

}
