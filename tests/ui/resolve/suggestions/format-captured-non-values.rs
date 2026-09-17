//! Captures must not suggest invalid field access or edits to custom macro invocations.

//@ edition: 2021
#![allow(dead_code)]

macro_rules! generated_arg {
    ($fmt:literal) => { format!($fmt, 1) };
}

macro_rules! forwarded_args {
    ($fmt:literal, $arg:expr) => { format!($fmt, $arg) };
}

#[derive(Debug)]
struct Example {
    value: usize,
}

impl Example {
    fn method(&self) {}

    fn report(&self) {
        let _ = format!("{method}");
        //~^ ERROR cannot find value `method` in this scope

        // Neither wrapper accepts another named argument, even when spans come from input.
        let _ = generated_arg!("{} {value}");
        //~^ ERROR cannot find value `value` in this scope
        let _ = forwarded_args!("{} {value}", 1);
        //~^ ERROR cannot find value `value` in this scope
    }
}

trait Report {
    type Item;
    fn limit() -> usize { 3 }
    fn report(&self) {
        let _ = format!("{Item}");
        //~^ ERROR cannot find value `Item` in this scope
    }
}

impl Report for () {
    type Item = ();

    fn report(&self) {
        // Label associated items instead of inserting `Self::` inside a capture.
        let _ = format!("{limit}");
        //~^ ERROR cannot find value `limit` in this scope
    }
}

impl std::fmt::Display for Example {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The associated-item definition may also come from another crate.
        let _ = format!("{fmt}");
        //~^ ERROR cannot find value `fmt` in this scope
        Ok(())
    }
}

fn main() {
    // Constructing a nonempty struct still needs placeholder field values.
    let _ = format!("{Example:?}");
    //~^ ERROR cannot find value `Example` in this scope
}
