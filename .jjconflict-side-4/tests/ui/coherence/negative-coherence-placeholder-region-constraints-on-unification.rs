//@ revisions: explicit implicit
//@[implicit] check-pass

#![forbid(coherence_leak_check)]
#![feature(negative_impls, with_negative_coherence)]

pub trait Marker {}

#[cfg(implicit)]
impl<T: ?Sized> !Marker for &T {}

#[cfg(explicit)]
impl<'a, T: ?Sized + 'a> !Marker for &'a T {}

trait FnMarker {}

// Unifying these two impls below results in a `T: '!0` obligation
// that we shouldn't need to care about. Ideally, we'd treat that
// as an assumption when proving `&'!0 T: Marker`...
impl<T: ?Sized + Marker> FnMarker for fn(T) {}
impl<T: ?Sized> FnMarker for fn(&T) {}
//[explicit]~^ ERROR conflicting implementations of trait `FnMarker` for type `fn(&_)`
//[explicit]~| WARN the behavior may change in a future release

fn main() {}
