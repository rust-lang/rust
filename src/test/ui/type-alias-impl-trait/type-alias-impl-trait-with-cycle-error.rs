#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;
//~^ ERROR: unconstrained opaque type

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
