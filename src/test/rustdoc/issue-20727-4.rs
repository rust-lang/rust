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
// ignore-cross-compile

extern crate issue_20727;

// @has issue_20727_4/trait.Index.html
pub trait Index<Idx: ?Sized> {
    // @has - '//*[@class="rust trait"]' 'trait Index<Idx: ?Sized> {'
    // @has - '//*[@class="rust trait"]' 'type Output: ?Sized'
    type Output: ?Sized;

    // @has - '//*[@class="rust trait"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    fn index(&self, index: Idx) -> &Self::Output;
}

// @has issue_20727_4/trait.IndexMut.html
pub trait IndexMut<Idx: ?Sized>: Index<Idx> {
    // @has - '//*[@class="rust trait"]' \
    //        'trait IndexMut<Idx: ?Sized>: Index<Idx> {'
    // @has - '//*[@class="rust trait"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output;
}

pub mod reexport {
    // @has issue_20727_4/reexport/trait.Index.html
    // @has - '//*[@class="rust trait"]' 'trait Index<Idx> where Idx: ?Sized {'
    // @has - '//*[@class="rust trait"]' 'type Output: ?Sized'
    // @has - '//*[@class="rust trait"]' \
    //        'fn index(&self, index: Idx) -> &Self::Output'
    pub use issue_20727::Index;

    // @has issue_20727_4/reexport/trait.IndexMut.html
    // @has - '//*[@class="rust trait"]' \
    //        'trait IndexMut<Idx>: Index<Idx> where Idx: ?Sized {'
    // @has - '//*[@class="rust trait"]' \
    //        'fn index_mut(&mut self, index: Idx) -> &mut Self::Output;'
    pub use issue_20727::IndexMut;
}
