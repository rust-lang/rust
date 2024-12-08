#![deny(unreachable_patterns)] //~ NOTE the lint level is defined here
#![allow(non_snake_case, non_upper_case_globals)]
mod x {
    pub use std::env::consts::ARCH;
    const X: i32 = 0; //~ NOTE there is a constant of the same name
}
fn main() {
    let input: i32 = 42;

    const god: i32 = 1;
    const GOOD: i32 = 1;
    const BAD: i32 = 2;

    let name: i32 = 42; //~ NOTE there is a binding of the same name

    match input {
        X => {} //~ NOTE matches any value
        _ => {} //~ ERROR unreachable pattern
        //~^ NOTE no value can reach this
    }
    match input {
        GOD => {} //~ HELP you might have meant to pattern match against the value of similarly named constant `god`
        //~^ NOTE matches any value
        _ => {} //~ ERROR unreachable pattern
        //~^ NOTE no value can reach this
    }
    match input {
        GOOOD => {} //~ HELP you might have meant to pattern match against the value of similarly named constant `GOOD`
        //~^ NOTE matches any value
        _ => {} //~ ERROR unreachable pattern
        //~^ NOTE no value can reach this
    }
    match input {
        name => {}
        //~^ NOTE matches any value
        _ => {} //~ ERROR unreachable pattern
        //~^ NOTE no value can reach this
    }
    match "" {
        ARCH => {} //~ HELP you might have meant to pattern match against the value of constant `ARCH`
        //~^ NOTE matches any value
        _ => {} //~ ERROR unreachable pattern
        //~^ NOTE no value can reach this
    }
}
