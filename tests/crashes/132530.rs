//@ known-bug: #132530

#![feature(non_lifetime_binders)]

trait Trait<'a, A> {
    type Assoc<'a> = i32;
}

fn a() -> impl for<T> Trait<Assoc = impl Trait<T>> {}
