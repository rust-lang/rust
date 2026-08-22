//@ compile-flags: -Znext-solver=globally

#![feature(non_lifetime_binders)]
#![allow(incomplete_features, bare_trait_objects)]

fn trivial<A: ?Sized>()
where
    for<B> Fn(A, B): Fn(A, A) + 'static,
    //~^ ERROR the size for values of type `A` cannot be known at compilation time
    //~| ERROR the size for values of type `A` cannot be known at compilation time
{
}

fn caller<T>() {
    trivial();
    //~^ ERROR expected an `Fn(_, _)` closure
    //~| ERROR type mismatch resolving `<dyn Fn(_, B) as FnOnce<(_, _)>>::Output == ()`
}

fn main() {
    trivial();
    //~^ ERROR expected an `Fn(_, _)` closure
    //~| ERROR type mismatch resolving `<dyn Fn(_, B) as FnOnce<(_, _)>>::Output == ()`
}
