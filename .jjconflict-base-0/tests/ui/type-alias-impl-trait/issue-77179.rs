// Regression test for the ICE in #77179.

#![feature(type_alias_impl_trait)]

type Pointer<T> = impl std::ops::Deref<Target = T>;

#[define_opaque(Pointer)]
fn test() -> Pointer<_> {
    //~^ ERROR the placeholder `_` is not allowed within types
    //~| ERROR expected generic type parameter, found `i32`
    Box::new(1)
}

fn main() {
    test();
}

extern "Rust" {
    fn bar() -> Pointer<_>;
    //~^ ERROR the placeholder `_` is not allowed within types
}
