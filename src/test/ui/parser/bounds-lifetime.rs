// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

type A = for<'a:> fn(); // OK
type A = for<'a:,> fn(); // OK
type A = for<'a> fn(); // OK
type A = for<> fn(); // OK
type A = for<'a: 'b + 'c> fn(); // OK (rejected later by ast_validation)
type A = for<'a: 'b,> fn(); // OK(rejected later by ast_validation)
type A = for<'a: 'b +> fn(); // OK (rejected later by ast_validation)
type A = for<'a, T> fn(); // OK (rejected later by ast_validation)
type A = for<,> fn(); //~ ERROR expected one of `>`, identifier, or lifetime, found `,`

fn main() {}
