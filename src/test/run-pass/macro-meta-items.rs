// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// ignore-pretty - token trees can't pretty print
// compile-flags: --cfg foo

#[feature(macro_rules)];

macro_rules! compiles_fine {
    ($at:meta) => {
        #[cfg($at)]
        static MISTYPED: () = "foo";
    }
}
macro_rules! emit {
    ($at:meta) => {
        #[cfg($at)]
        static MISTYPED: &'static str = "foo";
    }
}

// item
compiles_fine!(bar)
emit!(foo)

fn foo() {
    println!("{}", MISTYPED);
}

pub fn main() {
    // statement
    compiles_fine!(baz);
    emit!(baz);
    println!("{}", MISTYPED);
}

