//! regression test for issue <https://github.com/rust-lang/rust/issues/50581>
fn main() {
    |_: [_; continue]| {}; //~ ERROR: `continue` outside of a loop

    |_: [_; break]| (); //~ ERROR: `break` outside of a loop or labeled block

    while |_: [_; continue]| {} {} //~ ERROR: `continue` outside of a loop

    while |_: [_; break]| {} {} //~ ERROR: `break` outside of a loop

    loop {
        |_: [_; break]| {} //~ ERROR: `break` outside of a loop
    }

    loop {
        |_: [_; continue]| {} //~ ERROR: `continue` outside of a loop
    }
}
