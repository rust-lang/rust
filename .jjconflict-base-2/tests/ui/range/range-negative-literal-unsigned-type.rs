// Regression tests for: https://github.com/rust-lang/rust/issues/136514

#![allow(unreachable_patterns)]
fn main() {
    match 0u8 {
        -1..=2 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        -0..=0 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        -256..=2 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        -255..=2 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        0..=-1 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        -2..=-1 => {}
        //~^ ERROR the trait bound `u8: Neg` is not satisfied
        //~| ERROR the trait bound `u8: Neg` is not satisfied
        _ => {}
    }
}
