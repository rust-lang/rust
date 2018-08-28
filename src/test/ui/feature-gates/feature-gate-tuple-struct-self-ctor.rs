// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct ST(i32, i32);

impl ST {
    fn ctor() -> Self {
        Self(1,2)
        //~^ ERROR: expected function, found self type `Self` [E0423]
        //~^^ ERROR: tuple struct Self constructors are unstable (see issue #51994) [E0658]
    }
}
