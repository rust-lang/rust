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
use hash::{Hasher, SipHasher};
use rand::{mod, Rng};

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
// FIXME(#17307) the `H` type parameter should be an associated type
#[unstable = "interface may become more general in the future"]
pub trait HashState<O, H: Hasher<O>> {
    /// Creates a new hasher based on the given state of this object.
    fn hasher(&self) -> H;
}

/// A structure which is a factory for instances of `Hasher` which implement the
/// default trait.
///
/// This struct has is 0-sized and does not need construction.
pub struct DefaultState<Hasher>;

impl<O, T: Default + Hasher<O>> HashState<O, T> for DefaultState<T> {
    fn hasher(&self) -> T { Default::default() }
}

impl<T> Clone for DefaultState<T> {
    fn clone(&self) -> DefaultState<T> { DefaultState }
}

impl<T> Default for DefaultState<T> {
    fn default() -> DefaultState<T> { DefaultState }
}

/// `RandomSipState` is a factory for instances of `SipHasher` which is created
/// with random keys.
#[deriving(Clone)]
#[unstable = "this structure may not make it through to stability at 1.0"]
#[allow(missing_copy_implementations)]
pub struct RandomSipState {
    k0: u64,
    k1: u64,
}

impl RandomSipState {
    /// Construct a new `RandomSipState` that is initialized with random keys.
    #[inline]
    pub fn new() -> RandomSipState {
        let mut r = rand::task_rng();
        RandomSipState { k0: r.gen(), k1: r.gen() }
    }
}

impl HashState<u64, SipHasher> for RandomSipState {
    fn hasher(&self) -> SipHasher { SipHasher::new_with_keys(self.k0, self.k1) }
}

#[stable]
impl Default for RandomSipState {
    #[inline]
    fn default() -> RandomSipState {
        RandomSipState::new()
    }
}
