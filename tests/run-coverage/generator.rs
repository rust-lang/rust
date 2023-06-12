#![feature(generators, generator_trait)]

use std::ops::{Generator, GeneratorState};
use std::pin::Pin;

// The following implementation of a function called from a `yield` statement
// (apparently requiring the Result and the `String` type or constructor)
// creates conditions where the `generator::StateTransform` MIR transform will
// drop all `Counter` `Coverage` statements from a MIR. `simplify.rs` has logic
// to handle this condition, and still report dead block coverage.
fn get_u32(val: bool) -> Result<u32, String> {
    if val { Ok(1) } else { Err(String::from("some error")) }
}

fn main() {
    let is_true = std::env::args().len() == 1;
    let mut generator = || {
        yield get_u32(is_true);
        return "foo";
    };

    match Pin::new(&mut generator).resume(()) {
        GeneratorState::Yielded(Ok(1)) => {}
        _ => panic!("unexpected return from resume"),
    }
    match Pin::new(&mut generator).resume(()) {
        GeneratorState::Complete("foo") => {}
        _ => panic!("unexpected return from resume"),
    }
}
