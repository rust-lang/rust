//! Regression test for https://github.com/rust-lang/rust/issues/10638

//@ run-pass

pub fn main() {
    //// I am not a doc comment!
    ////////////////// still not a doc comment
    /////**** nope, me neither */
    /*** And neither am I! */
    5;
    /*****! certainly not I */
}
