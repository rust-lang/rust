//@ aux-build:other.rs

extern crate other;

fn main() {
    other::pair!(u8);
    //~^ ERROR pair! is missing its second argument
    //~| NOTE add the missing value here
    //~| NOTE while trying to match `,`
    //~| NOTE this macro expects a type and a value, like `pair!(u8, 0)`
}
