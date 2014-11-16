// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const FOO: &'static [u8, ..3] = b"foo";
const BAR: &'static [u8] = b"foo";

fn main() {
    let foo: &'static [u8, ..3] = b"foo";
    let bar: &'static [u8] = b"foo";
}
