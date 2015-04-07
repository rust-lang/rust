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

extern crate issue_20727;

pub trait Bar {}

// @has issue_20727_3/trait.Deref2.html
pub trait Deref2 {
    // @has - '//*[@class="rust trait"]' 'trait Deref2 {'
    // @has - '//*[@class="rust trait"]' 'type Target: Bar;'
    type Target: Bar;

    // @has - '//*[@class="rust trait"]' 'fn deref(&self) -> Self::Target;'
    fn deref(&self) -> Self::Target;
}

// @has issue_20727_3/reexport/trait.Deref2.html
pub mod reexport {
    // @has - '//*[@class="rust trait"]' 'trait Deref2 {'
    // @has - '//*[@class="rust trait"]' 'type Target: Bar;'
    // @has - '//*[@class="rust trait"]' 'fn deref(&self) -> Self::Target;'
    pub use issue_20727::Deref2;
}
