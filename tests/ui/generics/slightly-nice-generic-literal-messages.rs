use std::marker;

struct Foo<T,U>(T, marker::PhantomData<U>);

fn main() {
    match Foo(1.1, marker::PhantomData) { //~ NOTE this expression has type `Foo<{float}, _>`
        1 => {}
    //~^ ERROR mismatched types
    //~| NOTE expected struct `Foo<{float}, _>`
    //~| NOTE found type `{integer}`
    //~| NOTE expected `Foo<{float}, _>`, found integer
    }

}
