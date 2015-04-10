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

// @has issue_20727_2/trait.Add.html
pub trait Add<RHS = Self> {
    // @has - '//*[@class="rust trait"]' 'trait Add<RHS = Self> {'
    // @has - '//*[@class="rust trait"]' 'type Output;'
    type Output;

    // @has - '//*[@class="rust trait"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    fn add(self, rhs: RHS) -> Self::Output;
}

// @has issue_20727_2/reexport/trait.Add.html
pub mod reexport {
    // @has - '//*[@class="rust trait"]' 'trait Add<RHS = Self> {'
    // @has - '//*[@class="rust trait"]' 'type Output;'
    // @has - '//*[@class="rust trait"]' 'fn add(self, rhs: RHS) -> Self::Output;'
    pub use issue_20727::Add;
}

