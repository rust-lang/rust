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
pub use binary_heap::BinaryHeap;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use btree_map::BTreeMap;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use btree_set::BTreeSet;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use linked_list::LinkedList;

#[stable(feature = "rust1", since = "1.0.0")]
#[doc(no_inline)]
pub use vec_deque::VecDeque;

use crate::alloc::{Layout, LayoutError};
use core::fmt::Display;

/// The error type for `try_reserve` methods.
#[derive(Clone, PartialEq, Eq, Debug)]
#[unstable(feature = "try_reserve", reason = "new API", issue = "48043")]
pub enum TryReserveError {
    /// Error due to the computed capacity exceeding the collection's maximum
    /// (usually `isize::MAX` bytes).
    CapacityOverflow,

    /// The memory allocator returned an error
    AllocError {
        /// The layout of allocation request that failed
        layout: Layout,

        #[doc(hidden)]
        #[unstable(
            feature = "container_error_extra",
            issue = "none",
            reason = "\
            Enable exposing the allocator’s custom error value \
            if an associated type is added in the future: \
            https://github.com/rust-lang/wg-allocators/issues/23"
        )]
        non_exhaustive: (),
    },
}

#[unstable(feature = "try_reserve", reason = "new API", issue = "48043")]
impl From<LayoutError> for TryReserveError {
    #[inline]
    fn from(_: LayoutError) -> Self {
        TryReserveError::CapacityOverflow
    }
}

#[unstable(feature = "try_reserve", reason = "new API", issue = "48043")]
impl Display for TryReserveError {
    fn fmt(
        &self,
        fmt: &mut core::fmt::Formatter<'_>,
    ) -> core::result::Result<(), core::fmt::Error> {
        fmt.write_str("memory allocation failed")?;
        let reason = match &self {
            TryReserveError::CapacityOverflow => {
                " because the computed capacity exceeded the collection's maximum"
            }
            TryReserveError::AllocError { .. } => " because the memory allocator returned a error",
        };
        fmt.write_str(reason)
    }
}

/// An intermediate trait for specialization of `Extend`.
#[doc(hidden)]
trait SpecExtend<I: IntoIterator> {
    /// Extends `self` with the contents of the given iterator.
    fn spec_extend(&mut self, iter: I);
}
