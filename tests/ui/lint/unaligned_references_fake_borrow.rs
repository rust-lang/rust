//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/137250>.

// Ensure that we don't emit unaligned packed field reference errors for the fake
// borrows that we generate during match lowering. These fake borrows are there to
// ensure in *borrow-checking* that we don't modify the value being matched, but
// they are removed after the MIR is processed by `CleanupPostBorrowck`.

#[repr(packed)]
pub struct Packed(i32);

fn f(x: Packed) {
    match &x {
        Packed(4) => {},
        _ if true => {},
        _ => {}
    }

    match x {
        Packed(4) => {},
        _ if true => {},
        _ => {}
    }
}

fn main() {}
