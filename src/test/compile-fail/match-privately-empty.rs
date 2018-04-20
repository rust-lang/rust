// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]
#![feature(exhaustive_patterns)]

mod private {
    pub struct Private {
        _bot: !,
        pub misc: bool,
    }
    pub const DATA: Option<Private> = None;
}

fn main() {
    match private::DATA {
    //~^ ERROR non-exhaustive patterns: `Some(Private { misc: true, .. })` not covered
        None => {}
        Some(private::Private {
            misc: false,
            ..
        }) => {}
    }
}
