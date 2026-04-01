//@ build-fail
//@ compile-flags: -C debug-assertions

// This function is checking that our (type-based) automatic
// truncation does not sidestep the overflow checking.

#![deny(arithmetic_overflow)]

fn main() {
    // this signals overflow when checking is on
    let x = 2_i8 >> 17;
    //~^ ERROR: this arithmetic operation will overflow

    // ... but when checking is off, the fallback will truncate the
    // input to its lower three bits (= 1). Note that this is *not*
    // the behavior of the x86 processor for 8- and 16-bit types,
    // but it is necessary to avoid undefined behavior from LLVM.
    //
    // We check that here, by ensuring the result is not zero; if
    // overflow checking is turned off, then this assertion will pass
    // (and the compiletest driver will report that the test did not
    // produce the error expected above).
    assert_eq!(x, 1_i8);
}
