//! Regression test for https://github.com/rust-lang/rust/issues/162241.

fn main() {
    let never: Option<!> = None;
    let _: Option<u32> = never;
    //~^ ERROR mismatched types

    let x: Option<!> = None;
    let _: ! = x;
    //~^ ERROR mismatched types

    let never: Option<!> = loop {};
    let number: Option<u32> = never;
    //~^ ERROR mismatched types
}

fn question_mark() -> Option<u32> {
    let never: Option<!> = None;
    let _: u32 = never;
    //~^ ERROR mismatched types
    Some(0)
}
