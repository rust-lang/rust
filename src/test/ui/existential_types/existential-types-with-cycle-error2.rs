#![feature(existential_type)]

pub trait Bar<T> {
    type Item;
}

existential type Foo: Bar<Foo, Item = Foo>;
//~^ ERROR: could not find defining uses

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
