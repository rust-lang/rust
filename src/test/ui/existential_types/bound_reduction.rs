// build-pass (FIXME(62277): could be check-pass?)

#![allow(warnings)]

#![feature(existential_type)]

fn main() {
}

existential type Foo<V>: std::fmt::Debug;

trait Trait<U> {}

fn foo_desugared<T: Trait<[u32; {
    #[no_mangle]
    static FOO: usize = 42;
    3
}]>>(_: T) -> Foo<T> {
    (42, std::marker::PhantomData::<T>)
}
