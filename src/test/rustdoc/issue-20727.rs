// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-20727.rs
// ignore-android

extern crate issue_20727;

// @has issue_20727/trait.Deref.html
pub trait Deref {
    // @has - '//*[@class="rust trait"]' 'trait Deref {'
    // @has - '//*[@class="rust trait"]' 'type Target: ?Sized;'
    type Target: ?Sized;

    // @has - '//*[@class="rust trait"]' \
    //        "fn deref<'a>(&'a self) -> &'a Self::Target;"
    fn deref<'a>(&'a self) -> &'a Self::Target;
}

// @has issue_20727/reexport/trait.Deref.html
pub mod reexport {
    // @has - '//*[@class="rust trait"]' 'trait Deref {'
    // @has - '//*[@class="rust trait"]' 'type Target: ?Sized;'
    // @has - '//*[@class="rust trait"]' \
    //      "fn deref(&'a self) -> &'a Self::Target;"
    pub use issue_20727::Deref;
}
