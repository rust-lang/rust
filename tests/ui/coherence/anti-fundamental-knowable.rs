//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// `#[rustc_anti_fundamental]` restricts who may write an impl, not who owns `Self`, so coherence
// must still treat trait refs like `Box<dyn Signer>: Deref` or `Pin<LocalType>: DerefMut` as
// knowable when checking impl overlap.

#![feature(negative_impls)]

use std::ops::{Deref, DerefMut};
use std::pin::Pin;

pub trait Signer {
    fn pubkey(&self) -> u32;
}

impl<T> From<T> for Box<dyn Signer>
where
    T: Signer + 'static,
{
    fn from(signer: T) -> Self {
        Box::new(signer)
    }
}

impl<Container: Deref<Target = impl Signer>> Signer for Container {
    fn pubkey(&self) -> u32 {
        self.deref().pubkey()
    }
}

struct LocalType;

struct LocalPinned;
impl !Unpin for LocalPinned {}

trait DisjointFromDeref {}
impl<T: Deref> DisjointFromDeref for T {}
impl DisjointFromDeref for Pin<LocalType> {}

trait DisjointFromDerefMut {}
impl<T: DerefMut> DisjointFromDerefMut for T {}
impl DisjointFromDerefMut for Pin<LocalType> {}
impl DisjointFromDerefMut for Pin<&LocalType> {}
impl DisjointFromDerefMut for Pin<&mut LocalPinned> {}

// Nested fundamental types are knowable as well.
trait DisjointFromDerefNested {}
impl<T: Deref> DisjointFromDerefNested for T {}
impl DisjointFromDerefNested for Pin<Pin<LocalType>> {}

trait DisjointFromDerefMutNested {}
impl<T: DerefMut> DisjointFromDerefMutNested for T {}
impl DisjointFromDerefMutNested for &'static Pin<&'static LocalType> {}
impl DisjointFromDerefMutNested for Pin<Box<LocalPinned>> {}

fn main() {}
