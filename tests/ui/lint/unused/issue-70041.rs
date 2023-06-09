// compile-flags: --edition=2018
// run-pass

macro_rules! regex {
    //~^ WARN unused macro definition
    () => {};
}

#[allow(dead_code)]
use regex;
//~^ WARN unused import

fn main() {}
