// Because of #109628, when we compute the implied bounds from `Foo<X>`,
// we incorrectly get `X: placeholder('x)`.
// Make sure we ignore these bogus bounds and not use them for anything useful.
//
//@ revisions: fail pass
//@ [fail] check-fail
//@ [pass] check-pass

trait Trait {
    type Ty<'a> where Self: 'a;
}

impl<T> Trait for T {
    type Ty<'a> = () where Self: 'a;
}

struct Foo<T: Trait>(T)
where
    for<'x> T::Ty<'x>: Sized;

trait AnotherTrait {
    type Ty2<'a>: 'a;
}

#[cfg(fail)]
mod fail {
    use super::*;

    // implied_bound: `'lt: placeholder('x)`.
    // don't use the bound to prove `'lt: 'static`.
    fn test_lifetime<'lt, T: Trait>(_: Foo<&'lt u8>) {}
    //[fail]~^ ERROR `&'lt u8` does not fulfill the required lifetime
    //[fail]~| ERROR `&'lt u8` does not fulfill the required lifetime
    //[fail]~| ERROR may not live long enough

    // implied bound: `T::Ty2<'lt>: placeholder('x)`.
    // don't use the bound to prove `T::Ty2<'lt>: 'static`.
    fn test_alias<'lt, T: AnotherTrait>(_: Foo<T::Ty2::<'lt>>) {}
    //[fail]~^ ERROR `<T as AnotherTrait>::Ty2<'lt>` does not fulfill the required lifetime
    //[fail]~| ERROR `<T as AnotherTrait>::Ty2<'lt>` does not fulfill the required lifetime
    //[fail]~| ERROR may not live long enough
}


mod pass {
    use super::*;

    // implied_bound: 'static: placeholder('x).
    // don't ice.
    fn test_lifetime<T: Trait>(_: Foo<&'static u8>) {}

    // implied bound: T::Ty2<'static>: placeholder('x).
    // don't add the bound to the environment,
    // otherwise we would fail to infer a value for `'_`.
    fn test_alias<T: AnotherTrait>(_: Foo<T::Ty2::<'static>>) {
        None::<&'static T::Ty2<'_>>;
    }
}

fn main() {}
