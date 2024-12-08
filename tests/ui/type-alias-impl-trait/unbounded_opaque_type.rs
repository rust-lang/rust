//@ check-pass

#![feature(type_alias_impl_trait)]

mod opaque {
    pub type Opaque<T> = impl Sized;
    fn defining<T>() -> Opaque<T> {}
}

use opaque::Opaque;

struct Ss<'a, T>(&'a Opaque<T>);

fn test<'a, T>(_: Ss<'a, T>) {
    // test that we have an implied bound `Opaque<T>: 'a` from fn signature
    None::<&'a Opaque<T>>;
}

fn main() {}
