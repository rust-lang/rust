//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@check-pass
#![feature(try_as_dyn)]

trait Trait {}

impl<'a> Trait for &'a [(); 1] {}

impl Trait for &() {}
impl Trait for () {}

// Not fully generic impl -> returns None even tho
// implemented for *some* lifetimes
impl<'a> Trait for (&'a (), &'a ()) {}

// Not fully generic impl -> returns None even tho
// implemented for *some* lifetimes
impl<'a, 'b: 'a> Trait for (&'a (), &'b (), ()) {}

// Only valid for 'static lifetimes -> returns None
impl Trait for &'static u32 {}

trait Trait2 {}

struct Struct<T>(T);

// While this is the impl for `Trait`, in `Reflection` solver mode
// we reject the impl for `Trait2` below, and thus this impl also
// doesn't match.
impl<T: Trait2> Trait for Struct<T> {}

impl Trait2 for &'static u32 {}

// Test that downcasting to a dyn trait works as expected
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&42_i32).is_none());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&()).is_some());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&[()]).is_some());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&()).is_some());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&42_u32).is_none());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&(&(), &())).is_none());
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&(&(), &(), ())).is_none());

    assert!(std::any::try_as_dyn::<_, dyn Trait>(&Struct(&42_u32)).is_none());
};

fn main() {}
