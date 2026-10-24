//@ build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
#![feature(type_alias_impl_trait)]

fn main() {
}

type Foo<V> = impl std::fmt::Debug;

trait Trait<U> {}

#[define_opaque(Foo)]
fn foo_desugared<T: Trait<[u32; {
    #[no_mangle]
    static FOO: usize = 42;
    3
}]>>(_: T) -> Foo<T> {
    (42, std::marker::PhantomData::<T>)
}
