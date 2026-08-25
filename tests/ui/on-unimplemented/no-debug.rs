//@ aux-build:no_debug.rs

extern crate no_debug;

use no_debug::Bar;

struct Foo;

fn main() {
    println!("{:?} {:?}", Foo, Bar);
    println!("{} {}", Foo, Bar);
}
//~^^^ ERROR `Foo` doesn't implement `Debug`
//~| ERROR `Bar` doesn't implement `Debug`
//~^^^^ ERROR `Foo` doesn't implement `std::fmt::Display`
//~| ERROR `Bar` doesn't implement `std::fmt::Display`
