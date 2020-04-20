// run-rustfix
// check-pass

#![warn(rust_2018_compatibility)]

fn main() {
    try();
    //~^ WARNING `try` is a keyword in the 2018 edition
    //~| WARNING it will become a hard error in the 2018 edition!
}

fn try() {
    //~^ WARNING `try` is a keyword in the 2018 edition
    //~| WARNING it will become a hard error in the 2018 edition!
}
