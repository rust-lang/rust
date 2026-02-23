//! regression test for <https://github.com/rust-lang/rust/issues/27008>

struct S;

fn main() {
    let b = [0; S];
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `S`
}
