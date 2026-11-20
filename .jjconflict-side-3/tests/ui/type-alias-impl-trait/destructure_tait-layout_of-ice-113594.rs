//@ build-pass
//@ edition: 2021

#![feature(type_alias_impl_trait)]

fn foo<T>(x: T) {
    type Opaque<T> = impl Sized;
    let foo: Opaque<T> = (x,);
    let (a,): (T,) = foo;
}

const fn bar<T: Copy>(x: T) {
    type Opaque<T: Copy> = impl Copy;
    let foo: Opaque<T> = (x, 2u32);
    let (a, b): (T, u32) = foo;
}

fn main() {
    foo::<u32>(1);
    bar::<u32>(1);
    const CONST: () = bar::<u32>(42u32);
    CONST
}
