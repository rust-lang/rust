//@ only-apple
//@ build-fail
//@ dont-check-compiler-stderr
//@ dont-check-compiler-stdout

// Regression test for <https://github.com/rust-lang/rust/issues/139744>.
// Functions in the dynamic library marked with no_mangle should not be GC-ed.

#![crate_type = "cdylib"]

unsafe extern "C" {
    unsafe static THIS_SYMBOL_SHOULD_BE_UNDEFINED: usize;
}

#[unsafe(no_mangle)]
pub unsafe fn function_marked_with_no_mangle() {
    println!("FUNCTION_MARKED_WITH_NO_MANGLE = {}", unsafe { THIS_SYMBOL_SHOULD_BE_UNDEFINED });
}

//~? ERROR linking
