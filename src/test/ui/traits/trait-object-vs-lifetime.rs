// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A few contrived examples where lifetime should (or should not) be parsed as an object type.
// Lifetimes parsed as types are still rejected later by semantic checks.

// compile-flags: -Z continue-parse-after-error

struct S<'a, T>(&'a u8, T);

fn main() {
    // `'static` is a lifetime argument, `'static +` is a type argument
    let _: S<'static, u8>;
    let _: S<'static, 'static +>;
    //~^ at least one non-builtin trait is required for an object type
    let _: S<'static, 'static>;
    //~^ ERROR wrong number of lifetime parameters: expected 1, found 2
    //~| ERROR wrong number of type arguments: expected 1, found 0
    let _: S<'static +, 'static>;
    //~^ ERROR lifetime parameters must be declared prior to type parameters
    //~| ERROR at least one non-builtin trait is required for an object type
}
