use std::marker;

struct Foo<T,U>(T, marker::PhantomData<U>);

fn main() {
    match Foo(1.1, marker::PhantomData) {
        1 => {}
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected struct `Foo<{float}, _>`
    //~| NOTE_NONVIRAL found type `{integer}`
    //~| NOTE_NONVIRAL expected `Foo<{float}, _>`, found integer
    }

}
