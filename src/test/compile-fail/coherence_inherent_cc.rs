// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:coherence_inherent_cc_lib.rs

// Tests that methods that implement a trait cannot be invoked
// unless the trait is imported.

extern crate coherence_inherent_cc_lib;

mod Import {
    // Trait is in scope here:
    use coherence_inherent_cc_lib::TheStruct;
    use coherence_inherent_cc_lib::TheTrait;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn();
    }
}

mod NoImport {
    // Trait is not in scope here:
    use coherence_inherent_cc_lib::TheStruct;

    fn call_the_fn(s: &TheStruct) {
        s.the_fn(); //~ ERROR does not implement any method in scope named `the_fn`
    }
}

fn main() {}
