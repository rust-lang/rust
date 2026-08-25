//@ aux-build:other.rs

extern crate other;

fn main() {
    other::pair!(u8);
    //~^ ERROR invalid arguments to pair macro invocation
    //~| NOTE expected a type and value here
    //~| NOTE while trying to match `,`
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
    //~| NOTE see the macro documentation for accepted forms
}
