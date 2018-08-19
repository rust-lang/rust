// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(usize)]
#[derive(Copy, Clone)]
enum Enum {
    A = 0,
}

union Foo {
    a: &'static u8,
    b: Enum,
}

// A pointer is guaranteed non-null
const BAD_ENUM: Enum = unsafe { Foo { a: &1 }.b};
//~^ ERROR this constant likely exhibits undefined behavior

fn main() {
}
