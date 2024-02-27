//@compile-flags: -Zmiri-disable-abi-check
// This feature is required to trigger the error using the "C" ABI.
#![feature(c_unwind)]

extern "C" {
    fn miri_start_unwind(payload: *mut u8) -> !;
}

fn main() {
    unsafe { miri_start_unwind(&mut 0) }
    //~^ ERROR: unwinding past a stack frame that does not allow unwinding
}
