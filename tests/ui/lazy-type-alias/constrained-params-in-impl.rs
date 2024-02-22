//@ check-pass

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Injective<T> = Local<T>;
struct Local<T>(T);

impl<T> Injective<T> {
    fn take(_: T) {}
}

trait Trait {
    type Out;
    fn produce() -> Self::Out;
}

impl<T: Default> Trait for Injective<T> {
    type Out = T;
    fn produce() -> Self::Out { T::default() }
}

fn main() {
    Injective::take(0);
    let _: String = Injective::produce();
    let _: bool = Local::produce();
}
