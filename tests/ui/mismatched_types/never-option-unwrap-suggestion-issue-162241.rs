//! Regression test for https://github.com/rust-lang/rust/issues/162241.

fn main() {
    let never: Option<!> = None;
    let _: Option<u32> = never;
    //~^ ERROR mismatched types
}
