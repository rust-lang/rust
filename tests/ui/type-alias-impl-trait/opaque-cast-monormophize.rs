// build-pass

#![feature(type_alias_impl_trait)]

fn foo<T: Copy>(x: T) {
    type Opaque<T: Copy> = impl Copy;
    let foo: &Opaque<T> = &(x, 2u32);
    let (a, b): (T, u32) = *foo;
}

fn main() {
    foo::<u32>(1);
}
