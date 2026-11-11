//@ dont-require-annotations: NOTE

#![feature(arbitrary_self_types, coerce_unsized, dispatch_from_dyn, unsize)]
#![feature(unsized_fn_params)]

// This tests a few edge-cases around `arbitrary_self_types`. Most specifically,
// it checks that the `ObjectCandidate` you get from method matching can't
// match a trait with the same DefId as a supertrait but a bad type parameter.

use std::marker::PhantomData;

mod internal {
    use std::ops::{CoerceUnsized, Deref, DispatchFromDyn};
    use std::marker::{PhantomData, Unsize};

    pub struct Smaht<T: ?Sized, MISC>(pub Box<T>, pub PhantomData<MISC>);

    impl<T: ?Sized, MISC> Deref for Smaht<T, MISC> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized + Unsize<U>, U: ?Sized, MISC> CoerceUnsized<Smaht<U, MISC>>
        for Smaht<T, MISC>
    {}
    impl<T: ?Sized + Unsize<U>, U: ?Sized, MISC> DispatchFromDyn<Smaht<U, MISC>>
        for Smaht<T, MISC>
    {}

    pub trait Foo: X<u32> {}
    pub trait X<T> {
        fn foo(self: Smaht<Self, T>) -> T;
    }

    impl X<u32> for () {
        fn foo(self: Smaht<Self, u32>) -> u32 {
            0
        }
    }

    pub trait Marker {}
    impl Marker for dyn Foo {}
    impl<T: Marker + ?Sized> X<u64> for T {
        fn foo(self: Smaht<Self, u64>) -> u64 {
            1
        }
    }

    impl Deref for dyn Foo {
        type Target = ();
        fn deref(&self) -> &() { &() }
    }

    impl Foo for () {}
}

pub trait FinalFoo {
    fn foo(&self) -> u8;
}

impl FinalFoo for () {
    fn foo(&self) -> u8 { 0 }
}

mod nuisance_foo {
    pub trait NuisanceFoo {
        fn foo(self);
    }

    impl<T: ?Sized> NuisanceFoo for T {
        fn foo(self) {}
    }
}


fn objectcandidate_impl() {
    let x: internal::Smaht<(), u32> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u32> = x;

    // This picks `<dyn internal::Foo as X<u32>>::foo` via `ObjectCandidate`.
    //
    // The `TraitCandidate` is not relevant because `X` is not in scope.
    let z = x.foo();

    // Observe the type of `z` is `u32`
    let _seetype: () = z; //~ ERROR mismatched types
    //~| NOTE expected `()`, found `u32`
}

fn traitcandidate_impl() {
    use internal::X;

    let x: internal::Smaht<(), u64> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u64> = x;

    // This picks `<dyn internal::Foo as X<u64>>::foo` via `TraitCandidate`.
    //
    // The `ObjectCandidate` does not apply, as it only applies to
    // `X<u32>` (and not `X<u64>`).
    let z = x.foo();

    // Observe the type of `z` is `u64`
    let _seetype: () = z; //~ ERROR mismatched types
    //~| NOTE expected `()`, found `u64`
}

fn traitcandidate_impl_with_nuisance() {
    use internal::X;
    use nuisance_foo::NuisanceFoo;

    let x: internal::Smaht<(), u64> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u64> = x;

    // This picks `<dyn internal::Foo as X<u64>>::foo` via `TraitCandidate`.
    //
    // The `ObjectCandidate` does not apply, as it only applies to
    // `X<u32>` (and not `X<u64>`).
    //
    // The NuisanceFoo impl has the same priority as the `X` impl,
    // so we get a conflict.
    let z = x.foo(); //~ ERROR multiple applicable items in scope
}


fn neither_impl() {
    let x: internal::Smaht<(), u64> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u64> = x;

    // This can't pick the `TraitCandidate` impl, because `Foo` is not
    // imported. However, this also can't pick the `ObjectCandidate`
    // impl, because it only applies to `X<u32>` (and not `X<u64>`).
    //
    // Therefore, neither of the candidates is applicable, and we pick
    // the `FinalFoo` impl after another deref, which will return `u8`.
    let z = x.foo();

    // Observe the type of `z` is `u8`
    let _seetype: () = z; //~ ERROR mismatched types
    //~| NOTE expected `()`, found `u8`
}

fn both_impls() {
    use internal::X;

    let x: internal::Smaht<(), u32> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u32> = x;

    // This can pick both the `TraitCandidate` and the `ObjectCandidate` impl.
    //
    // However, the `ObjectCandidate` is considered an "inherent candidate",
    // and therefore has priority over both the `TraitCandidate` as well as
    // any other "nuisance" candidate" (if present).
    let z = x.foo();

    // Observe the type of `z` is `u32`
    let _seetype: () = z; //~ ERROR mismatched types
    //~| NOTE expected `()`, found `u32`
}


fn both_impls_with_nuisance() {
    // Similar to the `both_impls` example, except with a nuisance impl to
    // make sure the `ObjectCandidate` indeed has a higher priority.

    use internal::X;
    use nuisance_foo::NuisanceFoo;

    let x: internal::Smaht<(), u32> = internal::Smaht(Box::new(()), PhantomData);
    let x: internal::Smaht<dyn internal::Foo, u32> = x;
    let z = x.foo();

    // Observe the type of `z` is `u32`
    let _seetype: () = z; //~ ERROR mismatched types
    //~| NOTE expected `()`, found `u32`
}

fn main() {
}
