// aux-build:no_debug.rs

extern crate no_debug;

use no_debug::Bar;

struct Foo;

fn main() {
    println!("{:?} {:?}", Foo, Bar);
    println!("{} {}", Foo, Bar);
}
//~^^^ ERROR `Foo` doesn't implement `std::fmt::Debug`
//~| ERROR `no_debug::Bar` doesn't implement `std::fmt::Debug`
//~^^^^ ERROR `Foo` doesn't implement `std::fmt::Display`
//~| ERROR `no_debug::Bar` doesn't implement `std::fmt::Display`
