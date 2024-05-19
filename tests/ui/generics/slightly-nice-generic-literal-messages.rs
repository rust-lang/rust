use std::marker;

struct Foo<T,U>(T, marker::PhantomData<U>);

fn main() {
    match Foo(1.1, marker::PhantomData) {
        1 => {}
    //~^ ERROR mismatched types
    //~| expected struct `Foo<{float}, _>`
    //~| found type `{integer}`
    //~| expected `Foo<{float}, _>`, found integer
    }

}
