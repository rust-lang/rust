//@ edition: 2024
//@ compile-flags: -Zunstable-options

struct Foo(u8);

fn main() {
    let Foo(mut a) = &Foo(0);
    a = &42;
    //~^ ERROR: mismatched types

    let Foo(mut a) = &mut Foo(0);
    a = &mut 42;
    //~^ ERROR: mismatched types
}
