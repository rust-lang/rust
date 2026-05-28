// https://github.com/rust-lang/rust/issues/148431

// This test is designed to hit a case where, thanks to the
// recursion limit, the where clause gets generated, but not
// used, because we run out of fuel.
//
// This results in a reverse index with nothing in it, which
// used to crash when we parsed it.
pub fn foobar1<A: T1<B>, B: T2<C>, C: T3<D>, D: T4<A>>(a: A) {}

pub trait T1<T: ?Sized> {}
pub trait T2<T: ?Sized> {}
pub trait T3<T: ?Sized> {}
pub trait T4<T: ?Sized> {}

// foobar1 is the version that worked at the time this test was written
// the rest are here to try to make the test at least a little more
// robust, in the sense that it actually tests the code and isn't magically
// fixed by the recursion limit changing
pub fn foobar2<A: U1<B>, B: U2<C>, C: U3<D>, D: U4<E>, E: U5<A>>(a: A) {}

pub trait U1<T: ?Sized> {}
pub trait U2<T: ?Sized> {}
pub trait U3<T: ?Sized> {}
pub trait U4<T: ?Sized> {}
pub trait U5<T: ?Sized> {}

pub fn foobar3<A: V1<B>, B: V2<C>, C: V3<D>, D: V4<E>, E: V5<F>, F: V6<A>>(a: A) {}

pub trait V1<T: ?Sized> {}
pub trait V2<T: ?Sized> {}
pub trait V3<T: ?Sized> {}
pub trait V4<T: ?Sized> {}
pub trait V5<T: ?Sized> {}
pub trait V6<T: ?Sized> {}

pub fn foobar4<A: W1<B>, B: W2<C>, C: W3<A>>(a: A) {}

pub trait W1<T: ?Sized> {}
pub trait W2<T: ?Sized> {}
pub trait W3<T: ?Sized> {}
