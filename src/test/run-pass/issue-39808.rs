// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unreachable_code)]

// Regression test for #39808. The type parameter of `Owned` was
// considered to be "unconstrained" because the type resulting from
// `format!` (`String`) was not being propagated upward, owing to the
// fact that the expression diverges.

use std::borrow::Cow;

fn main() {
    let _ = if false {
        Cow::Owned(format!("{:?}", panic!()))
    } else {
        Cow::Borrowed("")
    };
}
