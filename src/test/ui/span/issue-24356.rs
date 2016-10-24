// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #24356

// ignore-tidy-linelength

fn main() {
    {
        use std::ops::Deref;

        struct Thing(i8);

        /*
        // Correct impl
        impl Deref for Thing {
            type Target = i8;
            fn deref(&self) -> &i8 { &self.0 }
        }
        */

        // Causes ICE
        impl Deref for Thing {
            //~^ ERROR E0046
            //~| NOTE missing `Target` in implementation
            //~| NOTE `Target` from trait: `type Target;`
            fn deref(&self) -> i8 { self.0 }
        }

        let thing = Thing(72);

        *thing
    };
}
