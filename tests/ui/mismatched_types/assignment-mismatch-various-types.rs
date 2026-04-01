//! regression test for <https://github.com/rust-lang/rust/issues/3477>

fn main() {
    let x: u32 = ();
    //~^ ERROR mismatched types

    let _p: char = 100;
    //~^ ERROR mismatched types
}
