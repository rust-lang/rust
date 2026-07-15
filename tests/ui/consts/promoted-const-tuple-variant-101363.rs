//! Regression test for <https://github.com/rust-lang/rust/issues/101363>.
//! Borrowing the result of a block expression inside a tuple variant
//! constructor in a `const` used to fail with E0716; it is now promoted.

//@ check-pass

const OPTIONAL_SLICE_V1: Option<&'static [u8]> = Some(&{
    let array = [1, 2, 3];
    array
});

const OPTIONAL_SLICE_V2: Option<&'static [u8]> = Some {
    0: &{
        let array = [1, 2, 3];
        array
    },
};

fn main() {}
