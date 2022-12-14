//@compile-flags: -Cpanic=abort

//! Unwinding despite `-C panic=abort` is an error.

extern "Rust" {
    fn miri_start_panic(payload: *mut u8) -> !;
}

fn main() {
    unsafe {
        miri_start_panic(&mut 0); //~ ERROR: unwinding past a stack frame that does not allow unwinding
    }
}
