//@ compile-flags: -C opt-level=1

#![no_builtins]
#![crate_type = "lib"]

// Make sure that there are no attributes or even types attached to this delcaration.
// If there were, they could be used even for other call sites using a different
// item importing the same function!
// See <https://github.com/rust-lang/rust/issues/46188> for details.
#[allow(improper_ctypes)]
extern "C" {
    // CHECK: @malloc = external global [0 x i8]
    pub fn malloc(x: u64) -> &'static mut ();
}

// `malloc` needs to be referenced or else it will disappear entirely.
#[no_mangle]
pub fn use_it() -> usize {
    let m: unsafe extern "C" fn(_) -> _ = malloc;
    malloc as *const () as usize
}
