// Regression test for #77179.

#![feature(type_alias_impl_trait)]

type Pointer<T> = impl std::ops::Deref<Target=T>;

fn test() -> Pointer<_> {
    //~^ ERROR: the placeholder `_` is not allowed within types
    Box::new(1)
}

fn main() {
    test();
}
