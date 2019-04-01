#![unstable(feature = "needle", issue = "56345")]

//! The Needle API, support generalized searching on strings, arrays and more.
//!
//! This module provides traits to facilitate searching [`Needle`] in a [`Haystack`].
//!
//! [`Needle`]: trait.Needle.html
//! [`Haystack`]: trait.Haystack.html
//!
//! Haystacks
//! =========
//!
//! A *haystack* refers to any linear structure which can be split or sliced
//! into smaller, non-overlapping parts. Examples are strings and vectors.
//!
//! ```rust
//! let haystack: &str = "hello";       // a string slice (`&str`) is a haystack.
//! let (a, b) = haystack.split_at(4);  // it can be split into two strings.
//! let c = &a[1..3];                   // it can be sliced.
//! ```
//!
//! The minimal haystack which cannot be further sliced is called a *codeword*.
//! For instance, the codeword of a string would be a UTF-8 sequence. A haystack
//! can therefore be viewed as a consecutive list of codewords.
//!
//! The boundary between codewords can be addressed using an *index*. The
//! numbers 1, 3 and 4 in the snippet above are sample indices of a string. An
//! index is usually a `usize`.
//!
//! An arbitrary number may point outside of a haystack, or in the interior of a
//! codeword. These indices are invalid. A *valid index* of a certain haystack
//! would only point to the boundaries.

mod haystack;
mod needle;
pub mod ext;

pub use self::haystack::*;
pub use self::needle::*;
