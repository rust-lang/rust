// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we handle projection types which wind up important for
// resolving methods. This test was reduced from a larger example; the
// call to `foo()` at the end was failing to resolve because the
// winnowing stage of method resolution failed to handle an associated
// type projection.

#![feature(associated_types)]

trait Hasher {
    type Output;
    fn finish(&self) -> Self::Output;
}

trait Hash<H: Hasher> {
    fn hash(&self, h: &mut H);
}

trait HashState {
    type Wut: Hasher;
    fn hasher(&self) -> Self::Wut;
}

struct SipHasher;
impl Hasher for SipHasher {
    type Output = u64;
    fn finish(&self) -> u64 { 4 }
}

impl Hash<SipHasher> for int {
    fn hash(&self, h: &mut SipHasher) {}
}

struct SipState;
impl HashState for SipState {
    type Wut = SipHasher;
    fn hasher(&self) -> SipHasher { SipHasher }
}

struct Map<S> {
    s: S,
}

impl<S> Map<S>
    where S: HashState,
          <S as HashState>::Wut: Hasher<Output=u64>,
{
    fn foo<K>(&self, k: K) where K: Hash< <S as HashState>::Wut> {}
}

fn foo<K: Hash<SipHasher>>(map: &Map<SipState>) {
    map.foo(22);
}

fn main() {}

