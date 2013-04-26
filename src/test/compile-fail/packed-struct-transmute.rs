// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This assumes the packed and non-packed structs are different sizes.

// the error points to the start of the file, not the line with the
// transmute

// error-pattern: transmute called on types with different size

#[packed]
struct Foo {
    bar: u8,
    baz: uint
}

struct Oof {
    rab: u8,
    zab: uint
}

fn main() {
    let foo = Foo { bar: 1, baz: 10 };
    unsafe {
        let oof: Oof = cast::transmute(foo);
        debug!(oof);
    }
}
