#![deny(incorrect_macro_fragment_repetition)]

macro_rules! foo {
    ($($a:expr)*) => {};
    //~^ ERROR `$a:expr` is followed (through repetition) by itself, which is not allowed for
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {}
