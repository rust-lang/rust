// aux-build:macro-brackets.rs

extern crate macro_brackets as bar;
use bar::doit;

macro_rules! id {
    ($($t:tt)*) => ($($t)*)
}

#[doit]
id![static X: u32 = 'a';]; //~ ERROR: mismatched types


fn main() {}
