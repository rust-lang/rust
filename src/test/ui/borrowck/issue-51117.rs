// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #51117 in borrowck interaction with match
// default bindings. The borrow of `*bar` created by `baz` was failing
// to register as a conflict with `bar.take()`.

fn main() {
    let mut foo = Some("foo".to_string());
    let bar = &mut foo;
    match bar {
        Some(baz) => {
            bar.take(); //~ ERROR cannot borrow
            drop(baz);
        },
        None => unreachable!(),
    }
}
