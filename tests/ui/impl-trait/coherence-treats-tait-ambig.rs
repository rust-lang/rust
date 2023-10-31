// revisions: current next
//[next] compile-flags: -Ztrait-solver=next

#![feature(type_alias_impl_trait)]

type T = impl Sized;

struct Foo;

impl Into<T> for Foo {
//~^ ERROR conflicting implementations of trait `Into<T>` for type `Foo`
    fn into(self) -> T {
        Foo
    }
}

fn main() {
    let _: T = Foo.into();
}
