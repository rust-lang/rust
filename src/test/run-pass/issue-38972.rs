// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This issue tracks a regression (a new warning) without
// feature(never_type). When we make that the default, please
// remove this test.

enum Foo { }

fn make_foo() -> Option<Foo> { None }

#[deny(warnings)]
fn main() {
    match make_foo() {
        None => {},
        Some(_) => {}
    }
}
