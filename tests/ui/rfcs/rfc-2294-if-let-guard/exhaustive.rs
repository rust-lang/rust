#![allow(irrefutable_let_patterns)]

//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

fn match_option(x: Option<u32>) {
    match x {
        //~^ ERROR non-exhaustive patterns: `None` not covered
        Some(_) => {}
        None if let y = x => {}
    }
}

fn main() {
    let x = ();
    match x {
        //~^ ERROR non-exhaustive patterns: `()` not covered
        y if let z = y => {}
    }
}
