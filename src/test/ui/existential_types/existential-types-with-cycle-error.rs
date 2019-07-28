#![feature(existential_type)]

existential type Foo: Fn() -> Foo;
//~^ ERROR: could not find defining uses

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
