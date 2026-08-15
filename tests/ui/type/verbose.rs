//@ revisions:verbose normal
//@ [verbose]compile-flags:--verbose
#![crate_type = "lib"]

struct Foo<T, U> { x: T, y: U }
fn bar() {
    let _: Foo<u32, i32> = Foo::<i32, i32> { x: 0, y: 0 };
    //~^ ERROR mismatched types
    //[verbose]~| NOTE expected struct `Foo<u32, i32>`
    //[normal]~| NOTE expected struct `Foo<u32, _>`
    //~| NOTE expected `Foo<u32, i32>`
    //~| NOTE expected due to this
}
