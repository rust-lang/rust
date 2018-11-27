// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const FOO: impl Copy = 42;
//~^ ERROR `impl Trait` not allowed

static BAR: impl Copy = 42;
//~^ ERROR `impl Trait` not allowed

fn main() {
    let foo = impl Copy = 42;
//~^ ERROR expected expression, found keyword `impl`
    let foo: impl Copy = 42;
}
