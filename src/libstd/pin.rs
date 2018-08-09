// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Types which pin data to its location in memory
//!
//! It is sometimes useful to have objects that are guaranteed to not move,
//! in the sense that their placement in memory in consistent, and can thus be relied upon.
//!
//! A prime example of such a scenario would be building self-referencial structs,
//! since moving an object with pointers to itself will invalidate them,
//! which could cause undefined behavior.
//!
//! In order to prevent objects from moving, they must be *pinned*,
//! by wrapping the data in special pointer types, such as [`PinMut`] and [`PinBox`].
//! These restrict access to the underlying data to only be immutable by implementing [`Deref`],
//! unless the type implements the [`Unpin`] trait,
//! which indicates that it doesn't need these restrictions and can be safely mutated,
//! by implementing [`DerefMut`].
//!
//! This is done because, while modifying an object can be done in-place,
//! it might also relocate a buffer when its at full capacity,
//! or it might replace one object with another without logically "moving" them with [`swap`].
//!
//! [`PinMut`]: struct.PinMut.html
//! [`PinBox`]: struct.PinBox.html
//! [`Unpin`]: ../marker/trait.Unpin.html
//! [`DerefMut`]: ../ops/trait.DerefMut.html
//! [`Deref`]: ../ops/trait.Deref.html
//! [`swap`]: ../mem/fn.swap.html
//!
//! # Examples
//!
//! ```rust
//! #![feature(pin)]
//!
//! use std::pin::PinBox;
//! use std::marker::Pinned;
//! use std::ptr::NonNull;
//!
//! // This is a self referencial struct since the slice field points to the data field.
//! // We cannot inform the compiler about that with a normal reference,
//! // since this pattern cannot be described with the usual borrowing rules.
//! // Instead we use a raw pointer, though one which is known to not be null,
//! // since we know it's pointing at the string.
//! struct Unmovable {
//!     data: String,
//!     slice: NonNull<String>,
//!     _pin: Pinned,
//! }
//!
//! impl Unmovable {
//!     // To ensure the data doesn't move when the function returns,
//!     // we place it in the heap where it will stay for the lifetime of the object,
//!     // and the only way to access it would be through a pointer to it.
//!     fn new(data: String) -> PinBox<Self> {
//!         let res = Unmovable {
//!             data,
//!             // we only create the pointer once the data is in place
//!             // otherwise it will have already moved before we even started
//!             slice: NonNull::dangling(),
//!             _pin: Pinned,
//!         };
//!         let mut boxed = PinBox::new(res);
//!
//!         let slice = NonNull::from(&boxed.data);
//!         // we know this is safe because modifying a field doesn't move the whole struct
//!         unsafe { PinBox::get_mut(&mut boxed).slice = slice };
//!         boxed
//!     }
//! }
//!
//! let unmoved = Unmovable::new("hello".to_string());
//! // The pointer should point to the correct location,
//! // so long as the struct hasn't moved.
//! // Meanwhile, we are free to move the pointer around.
//! let mut still_unmoved = unmoved;
//! assert_eq!(still_unmoved.slice, NonNull::from(&still_unmoved.data));
//!
//! // Now the only way to access to data (safely) is immutably,
//! // so this will fail to compile:
//! // still_unmoved.data.push_str(" world");
//!
//! ```

#![unstable(feature = "pin", issue = "49150")]

pub use core::pin::*;

pub use alloc_crate::pin::*;
