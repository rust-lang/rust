// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this now fails (correctly, I claim) because hygiene prevents
// the assembled identifier from being a reference to the binding.
#![feature(concat_idents)]

pub fn main() {
    let asdf_fdsa = "<.<".to_owned();
    assert_eq!(concat_idents!(asd, f_f, dsa), "<.<".to_owned());
    //~^ ERROR: unresolved name `asdf_fdsa`

    assert!(stringify!(use_mention_distinction) ==
                "use_mention_distinction");
}
