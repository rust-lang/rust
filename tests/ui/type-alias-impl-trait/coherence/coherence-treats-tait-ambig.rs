#![feature(type_alias_impl_trait)]

type T = impl Sized;

struct Foo;

impl Into<T> for Foo {
    //~^ ERROR conflicting implementations of trait `Into<_>` for type `Foo`
    #[define_opaque(T)]
    fn into(self) -> T {
        Foo
    }
}

fn main() {
    let _: T = Foo.into();
}
