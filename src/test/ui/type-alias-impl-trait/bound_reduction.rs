// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]

fn main() {
}

type Foo<V> = impl std::fmt::Debug;

trait Trait<U> {}

fn foo_desugared<T: Trait<[u32; {
    #[no_mangle]
    static FOO: usize = 42;
    3
}]>>(_: T) -> Foo<T> {
    (42, std::marker::PhantomData::<T>)
}
