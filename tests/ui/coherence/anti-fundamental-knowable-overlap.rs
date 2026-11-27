//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Counterpart to `anti-fundamental-knowable.rs`: treating trait refs such as `Pin<..>: DerefMut`
// as knowable must not hide impls that actually overlap, nor cases where an upstream crate could
// still add an impl.

use std::ops::{Deref, DerefMut};
use std::pin::Pin;

struct LocalType;
trait LocalTrait {}

trait OverlapDerefMut {}
impl<T: DerefMut> OverlapDerefMut for T {}
impl OverlapDerefMut for Pin<&'static mut LocalType> {}
//~^ ERROR conflicting implementations of trait `OverlapDerefMut`

trait OverlapDeref {}
impl<T: Deref> OverlapDeref for T {}
impl OverlapDeref for Pin<Box<LocalType>> {}
//~^ ERROR conflicting implementations of trait `OverlapDeref`
impl OverlapDeref for Box<dyn LocalTrait> {}
//~^ ERROR conflicting implementations of trait `OverlapDeref`
impl OverlapDeref for &'static Pin<&'static LocalType> {}
//~^ ERROR conflicting implementations of trait `OverlapDeref`
impl OverlapDeref for Box<Pin<LocalType>> {}
//~^ ERROR conflicting implementations of trait `OverlapDeref`

trait OverlapDerefMutBox {}
impl<T: DerefMut> OverlapDerefMutBox for T {}
impl OverlapDerefMutBox for Box<Pin<LocalType>> {}
//~^ ERROR conflicting implementations of trait `OverlapDerefMutBox`

// `U` is not local, so `Pin<&mut U>: DerefMut` is not knowable.
trait NotKnowable {}
impl<T: DerefMut> NotKnowable for T {}
impl<U: 'static> NotKnowable for Pin<&'static mut U> {}
//~^ ERROR conflicting implementations of trait `NotKnowable`

fn main() {}
