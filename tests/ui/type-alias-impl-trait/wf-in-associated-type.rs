// WF check for impl Trait in associated type position.
//
//@ revisions: pass pass_next fail
//@ [pass] check-pass
//@ [pass_next] compile-flags: -Znext-solver
//@ [pass_next] check-pass
//@ [fail] check-fail

#![feature(impl_trait_in_assoc_type)]

// The hidden type here (`&'a T`) requires proving `T: 'a`.
// We know it holds because of implied bounds from the impl header.
#[cfg(any(pass, pass_next))]
mod pass {
    trait Trait<Req> {
        type Opaque1;
        fn constrain_opaque1(req: Req) -> Self::Opaque1;
    }

    impl<'a, T> Trait<&'a T> for () {
        type Opaque1 = impl IntoIterator<Item = impl Sized + 'a>;
        fn constrain_opaque1(req: &'a T) -> Self::Opaque1 {
            [req]
        }
    }
}

// The hidden type here (`&'a T`) requires proving `T: 'a`,
// but that is not known to hold in the impl.
#[cfg(fail)]
mod fail {
    trait Trait<'a, T> {
        type Opaque;
        fn constrain_opaque(req: &'a T) -> Self::Opaque;
    }

    impl<'a, T> Trait<'a, T> for () {
        type Opaque = impl Sized + 'a;
        fn constrain_opaque(req: &'a T) -> Self::Opaque {
            req
            //[fail]~^ ERROR the parameter type `T` may not live long enough
            //[fail]~| ERROR the parameter type `T` may not live long enough
        }
    }
}

fn main() {}
