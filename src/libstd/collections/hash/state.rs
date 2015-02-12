// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use clone::Clone;
use default::Default;
use hash;
use marker;

/// A trait representing stateful hashes which can be used to hash keys in a
/// `HashMap`.
///
/// A HashState is used as a factory for instances of `Hasher` which a `HashMap`
/// can then use to hash keys independently. A `HashMap` by default uses a state
/// which will create instances of a `SipHasher`, but a custom state factory can
/// be provided to the `with_hash_state` function.
///
/// If a hashing algorithm has no initial state, then the `Hasher` type for that
/// algorithm can implement the `Default` trait and create hash maps with the
/// `DefaultState` structure. This state is 0-sized and will simply delegate
/// to `Default` when asked to create a hasher.
#[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
pub trait HashState {
    type Hasher: hash::Hasher;

    /// Creates a new hasher based on the given state of this object.
    fn hasher(&self) -> Self::Hasher;
}

/// A structure which is a factory for instances of `Hasher` which implement the
/// default trait.
///
/// This struct has is 0-sized and does not need construction.
#[unstable(feature = "std_misc", reason = "hasher stuff is unclear")]
pub struct DefaultState<H>(marker::PhantomData<H>);

impl<H: Default + hash::Hasher> HashState for DefaultState<H> {
    type Hasher = H;
    fn hasher(&self) -> H { Default::default() }
}

impl<H> Clone for DefaultState<H> {
    fn clone(&self) -> DefaultState<H> { DefaultState(marker::PhantomData) }
}

impl<H> Default for DefaultState<H> {
    fn default() -> DefaultState<H> { DefaultState(marker::PhantomData) }
}
