// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// An iterator whose items are random accessible efficiently
///
/// # Safety
///
/// The iterator's .len() and size_hint() must be exact.
/// `.len()` must be cheap to call.
///
/// .get_unchecked() must return distinct mutable references for distinct
/// indices (if applicable), and must return a valid reference if index is in
/// 0..self.len().
#[doc(hidden)]
pub unsafe trait TrustedRandomAccess : ExactSizeIterator {
    unsafe fn get_unchecked(&mut self, i: usize) -> Self::Item;
    /// Returns `true` if getting an iterator element may have
    /// side effects. Remember to take inner iterators into account.
    fn may_have_side_effect() -> bool;
}
