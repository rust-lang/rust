#![feature(existential_type)]

existential type Foo: Fn() -> Foo;
//~^ ERROR: cycle detected when processing `Foo`

fn crash(x: Foo) -> Foo {
    x
}

fn main() {

}
