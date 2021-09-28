// edition:2018

// This is a regression test for #83564.
// For some reason, Rust 2018 or higher is required to reproduce the bug.

fn main() {
    //~^ HELP consider importing one of these items
    let _x = NonZeroU32::new(5).unwrap();
    //~^ ERROR failed to resolve: use of undeclared type `NonZeroU32`
}
