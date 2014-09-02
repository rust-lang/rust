
// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Previously failed formating invalid utf8.
// cc #16877

// error-pattern:failed at 'hello�'

struct Foo;
impl std::fmt::Show for Foo {
    fn fmt(&self, fmtr:&mut std::fmt::Formatter) -> std::fmt::Result {
        // Purge invalid utf8: 0xff
        fmtr.write(&[104, 101, 108, 108, 111, 0xff])
    }
}
fn main() {
    fail!("{}", Foo)
}
