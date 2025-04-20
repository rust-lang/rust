//@ build-pass

// Regression test for <https://github.com/rust-lang/rust/issues/139744>.
// Functions in the binary marked with no_mangle should be GC-ed if they
// are not indirectly referenced by main.

#![feature(used_with_arg)]

unsafe extern "C" {
    unsafe static THIS_SYMBOL_SHOULD_BE_UNDEFINED: usize;
}

#[unsafe(no_mangle)]
pub unsafe fn function_marked_with_no_mangle() {
    println!("FUNCTION_MARKED_WITH_NO_MANGLE = {}", unsafe { THIS_SYMBOL_SHOULD_BE_UNDEFINED });
}

#[used(compiler)]
pub static FUNCTION_MARKED_WITH_USED: unsafe fn() = || {
    println!("FUNCTION_MARKED_WITH_USED = {}", unsafe { THIS_SYMBOL_SHOULD_BE_UNDEFINED });
};

fn main() {
    println!("MAIN");
}
