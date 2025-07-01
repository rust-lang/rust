// Test that the correct error is emitted when `#[const_continue]` occurs outside of
// a loop match. See also https://github.com/rust-lang/rust/issues/143165.
#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

fn main() {
    loop {
        #[const_continue]
        break ();
        //~^ ERROR `#[const_continue]` must break to a labeled block that participates in a `#[loop_match]`
    }
}
