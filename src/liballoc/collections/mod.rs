// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Collection types.

#![stable(feature = "rust1", since = "1.0.0")]

pub mod binary_heap;
mod btree;
pub mod linked_list;
pub mod vec_deque;

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_map {
    //! A map based on a B-Tree.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::btree::map::*;
}

#[stable(feature = "rust1", since = "1.0.0")]
pub mod btree_set {
    //! A set based on a B-Tree.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub use super::btree::set::*;
}

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use self::binary_heap::BinaryHeap;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use self::btree_map::BTreeMap;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use self::btree_set::BTreeSet;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use self::linked_list::LinkedList;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use self::vec_deque::VecDeque;

/// An intermediate trait for specialization of `Extend`.
#[doc(hidden)]
trait SpecExtend<I: IntoIterator> {
    /// Extends `self` with the contents of the given iterator.
    fn spec_extend(&mut self, iter: I);
}
