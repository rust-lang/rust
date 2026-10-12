//@ compile-flags: --crate-type lib

trait Foo {
    fn hello<const N: i32>();
}

struct Bar;

impl Foo for Bar {
    //~^ ERROR: not all trait items implemented, missing: `hello`
    fn hello<const N: >() {
        //~^ ERROR: expected type, found `>`
        println!("woof woof")
    }
}
