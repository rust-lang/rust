#![feature(type_alias_impl_trait)]

type Foo = impl Fn() -> Foo;
//~^ ERROR: could not find defining uses

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
