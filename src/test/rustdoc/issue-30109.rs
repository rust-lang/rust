// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// build-aux-docs
// aux-build:issue-30109-1.rs
// ignore-cross-compile

pub mod quux {
    extern crate issue_30109_1 as bar;
    use self::bar::Bar;

    pub trait Foo {}

    // @has issue_30109/quux/trait.Foo.html \
    //          '//a/@href' '../issue_30109_1/struct.Bar.html'
    impl Foo for Bar {}
}
