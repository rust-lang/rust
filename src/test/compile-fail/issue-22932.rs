// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

// Issue 22932: `panic!("{}");` should not compile.

pub fn f1() { panic!("this does not work {}");
              //~^ WARN unary `panic!` literal argument contains `{`
              //~| NOTE Is it meant to be a `format!` string?
              //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn workaround_1() {
    panic!(("This *does* works {}"));
}

pub fn workaround_2() {
    const MSG: &'static str = "This *does* work {}";
    panic!(MSG);
}

pub fn f2() { panic!("this does not work {");
              //~^ WARN unary `panic!` literal argument contains `{`
              //~| NOTE Is it meant to be a `format!` string?
              //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn f3() { panic!("nor this }");
              //~^ WARN unary `panic!` literal argument contains `}`
              //~| NOTE Is it meant to be a `format!` string?
              //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn f4() { panic!("nor this {{");
              //~^ WARN unary `panic!` literal argument contains `{`
              //~| NOTE Is it meant to be a `format!` string?
              //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn f5() { panic!("nor this }}");
              //~^ WARN unary `panic!` literal argument contains `}`
              //~| NOTE Is it meant to be a `format!` string?
              //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn f0_a() {
    __unstable_rustc_ensure_not_fmt_string_literal!("`f0_a`", "this does not work {}");
    //~^ WARN `f0_a` literal argument contains `{`
    //~| NOTE Is it meant to be a `format!` string?
    //~| HELP You can wrap the argument in parentheses to sidestep this warning
}

pub fn f0_b() {
    __unstable_rustc_ensure_not_fmt_string_literal!("`f0_b`", "this does work");
}

pub fn f0_c() {
    __unstable_rustc_ensure_not_fmt_string_literal!("`f0_c`", ("so does this {}"));
}

// This test is just checking that we get all the right warnings; none
// of them are outright errors, so use the special `rustc_error`
// attribute to force a compile error.
#[rustc_error]
pub fn main() {
    //~^ ERROR compilation successful
}
