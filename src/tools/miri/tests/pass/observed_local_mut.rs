// Stacked Borrows catches this (correctly) as UB.
//@compile-flags: -Zmiri-disable-stacked-borrows

// This test is intended to guard against the problem described in commit
// 39bb1254d1eaf74f45a4e741097e33fc942168d5.
//
// As written, it might be considered UB in compiled Rust, but of course Miri gives it a safe,
// deterministic behaviour (one that might not correspond with how an eventual Rust spec would
// defined this).
//
// An alternative way to write the test without `unsafe` would be to use `Cell<i32>`, but it would
// only surface the bug described by the above commit if `Cell<i32>` on the stack got represented
// as a primitive `PrimVal::I32` which is not yet the case.

fn main() {
    let mut x = 0;
    let y: *const i32 = &x;
    x = 1;

    // When the described bug is in place, this results in `0`, not observing the `x = 1` line.
    assert_eq!(unsafe { *y }, 1);

    assert_eq!(x, 1);
}
