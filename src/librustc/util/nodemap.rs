// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An efficient hash map for node IDs

#![allow(non_snake_case)]

use std::collections::hash_state::{DefaultState};
use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::hash::{Hasher, Hash};
#[cfg(stage0)] use std::hash::Writer;
use syntax::ast;

pub type FnvHashMap<K, V> = HashMap<K, V, DefaultState<FnvHasher>>;
pub type FnvHashSet<V> = HashSet<V, DefaultState<FnvHasher>>;

pub type NodeMap<T> = FnvHashMap<ast::NodeId, T>;
pub type DefIdMap<T> = FnvHashMap<ast::DefId, T>;

pub type NodeSet = FnvHashSet<ast::NodeId>;
pub type DefIdSet = FnvHashSet<ast::DefId>;

#[cfg(stage0)]
pub fn FnvHashMap<K: Hash<FnvHasher> + Eq, V>() -> FnvHashMap<K, V> {
    Default::default()
}
#[cfg(stage0)]
pub fn FnvHashSet<V: Hash<FnvHasher> + Eq>() -> FnvHashSet<V> {
    Default::default()
}
#[cfg(not(stage0))]
pub fn FnvHashMap<K: Hash + Eq, V>() -> FnvHashMap<K, V> {
    Default::default()
}
#[cfg(not(stage0))]
pub fn FnvHashSet<V: Hash + Eq>() -> FnvHashSet<V> {
    Default::default()
}

pub fn NodeMap<T>() -> NodeMap<T> { FnvHashMap() }
pub fn DefIdMap<T>() -> DefIdMap<T> { FnvHashMap() }
pub fn NodeSet() -> NodeSet { FnvHashSet() }
pub fn DefIdSet() -> DefIdSet { FnvHashSet() }

/// A speedy hash algorithm for node ids and def ids. The hashmap in
/// libcollections by default uses SipHash which isn't quite as speedy as we
/// want. In the compiler we're not really worried about DOS attempts, so we
/// just default to a non-cryptographic hash.
///
/// This uses FNV hashing, as described here:
/// http://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
pub struct FnvHasher(u64);

impl Default for FnvHasher {
    fn default() -> FnvHasher { FnvHasher(0xcbf29ce484222325) }
}

#[cfg(stage0)]
impl Hasher for FnvHasher {
    type Output = u64;
    fn reset(&mut self) { *self = Default::default(); }
    fn finish(&self) -> u64 { self.0 }
}

#[cfg(stage0)]
impl Writer for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;
        for byte in bytes {
            hash = hash ^ (*byte as u64);
            hash = hash * 0x100000001b3;
        }
        *self = FnvHasher(hash);
    }
}

#[cfg(not(stage0))]
impl Hasher for FnvHasher {
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;
        for byte in bytes {
            hash = hash ^ (*byte as u64);
            hash = hash * 0x100000001b3;
        }
        *self = FnvHasher(hash);
    }
    fn finish(&self) -> u64 { self.0 }
}
