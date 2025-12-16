//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@check-pass
#![feature(try_as_dyn)]

trait Trait {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&42_i32).is_none());
};

impl<'a> Trait for &'a [(); 1] {}
const _: () = {
    let x = ();
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&[x]).is_some());
};

type Foo = &'static [(); 2];

// Ensure type aliases don't skip these checks
impl Trait for Foo {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&[(), ()]).is_none());
};

impl Trait for &() {}
const _: () = {
    let x = ();
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&x).is_some());
};

impl Trait for () {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&()).is_some());
};

// Not fully generic impl -> returns None even tho
// implemented for *some* lifetimes
impl<'a> Trait for (&'a (), &'a ()) {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&(&(), &())).is_none());
};

// Not fully generic impl -> returns None even tho
// implemented for *some* lifetimes
impl<'a, 'b: 'a> Trait for (&'a (), &'b (), ()) {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&(&(), &(), ())).is_none());
};

// Only valid for 'static lifetimes -> returns None
// even though we are actually using a `'static` lifetime.
// We can't know what lifetimes are there during codegen, so
// we pessimistically assume it could be a shorter one
impl Trait for &'static u32 {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&&42_u32).is_none());
};

trait Trait2 {}

struct Struct<T>(T);

// While this is the impl for `Trait`, in `Reflection` solver mode
// we reject the impl for `Trait2` below, and thus this impl also
// doesn't match.
impl<T: Trait2> Trait for Struct<T> {}

impl Trait2 for &'static u32 {}
const _: () = {
    assert!(std::any::try_as_dyn::<_, dyn Trait>(&Struct(&42_u32)).is_none());
};

const _: () = {
    trait Homo {}
    impl<T> Homo for (T, T) {}

    // Let's pick `T = &'_ i32`.
    assert!(std::any::try_as_dyn::<_, dyn Homo>(&(&42_i32, &27_i32)).is_none());
};

fn main() {}
