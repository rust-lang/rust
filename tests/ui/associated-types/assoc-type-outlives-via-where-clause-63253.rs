//! Regression test for <https://github.com/rust-lang/rust/issues/63253>.
//!
//! A `where Self::Ty: 'a` bound on the callee was not being used to prove the
//! associated type outlives `'a` at the call site, so both of these calls used
//! to fail with E0309 ("the associated type `<T as Trait<'_>>::Ty` may not live
//! long enough").

//@ check-pass

#![allow(unused)]

// The associated function is reached through a method-call path.
trait Trait<'a> {
    type Ty;
    fn method(ty_ref: &'a Self::Ty) where Self::Ty: 'a {}
}

fn caller<'a, T: Trait<'a>>(arg: &'a T::Ty) where T::Ty: 'a {
    T::method(arg)
}

// The same bound, reached through a free function instead.
trait Trait2<'a> {
    type Ty;
}

fn free_fn<'a, T: Trait2<'a>>(_arg: &'a T::Ty) where T::Ty: 'a {}

fn free_fn_caller<'a, T: Trait2<'a>>(arg: &'a T::Ty) where T::Ty: 'a {
    free_fn::<T>(arg)
}

fn main() {}
