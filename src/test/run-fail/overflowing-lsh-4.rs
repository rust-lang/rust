// error-pattern:thread 'main' panicked at 'attempt to shift left with overflow'
// compile-flags: -C debug-assertions

// This function is checking that our automatic truncation does not
// sidestep the overflow checking.

#![warn(exceeding_bitshifts)]

fn main() {
    // this signals overflow when checking is on
    let x = 1_i8 << 17;

    // ... but when checking is off, the fallback will truncate the
    // input to its lower three bits (= 1). Note that this is *not*
    // the behavior of the x86 processor for 8- and 16-bit types,
    // but it is necessary to avoid undefined behavior from LLVM.
    //
    // We check that here, by ensuring the result has only been
    // shifted by one place; if overflow checking is turned off, then
    // this assertion will pass (and the compiletest driver will
    // report that the test did not produce the error expected above).
    assert_eq!(x, 2_i8);
}
