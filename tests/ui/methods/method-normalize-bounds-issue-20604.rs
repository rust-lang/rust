//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(stable_features)]

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

impl Hash<SipHasher> for isize {
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
