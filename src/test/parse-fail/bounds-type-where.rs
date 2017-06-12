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

type A where for<'a> for<'b> Trait1 + ?Trait2: 'a + Trait = u8; // OK
type A where T: Trait, = u8; // OK
type A where T: = u8; // OK
type A where T:, = u8; // OK
type A where T: Trait + Trait = u8; // OK
type A where = u8; // OK
type A where T: Trait + = u8; // OK
type A where T, = u8;
//~^ ERROR expected one of `!`, `(`, `+`, `::`, `:`, `==`, or `=`, found `,`

fn main() {}
