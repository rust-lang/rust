//@ known-bug: #121637
#![feature(non_lifetime_binders)]
trait Trait<Type> {
    type Type;

    fn method(&self) -> impl for<T> Trait<impl Trait<T>>;
}
