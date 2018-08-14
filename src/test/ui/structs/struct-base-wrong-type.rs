// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that `base` in `Fru { field: expr, ..base }` must have right type.
//
// See also struct-base-wrong-type-2.rs, which tests same condition
// within a function body.

struct Foo { a: isize, b: isize }
struct Bar { x: isize }

static bar: Bar = Bar { x: 5 };
static foo: Foo = Foo { a: 2, ..bar }; //~  ERROR mismatched types
                                       //~| expected type `Foo`
                                       //~| found type `Bar`
                                       //~| expected struct `Foo`, found struct `Bar`
static foo_i: Foo = Foo { a: 2, ..4 }; //~  ERROR mismatched types
                                       //~| expected type `Foo`
                                       //~| found type `{integer}`
                                       //~| expected struct `Foo`, found integral variable

fn main() {
    let b = Bar { x: 5 };
    // errors below are no longer caught since error above causes
    // compilation to abort before we bother checking function bodies.
    // See also struct-base-wrong-type-2.rs, which checks that we
    // would catch these errors eventually.
    let f = Foo { a: 2, ..b };
    let f__isize = Foo { a: 2, ..4 };
}
