// compile-flags: -Zmiri-disable-abi-check
#![feature(unwind_attributes, c_unwind)]

#[no_mangle]
extern "C-unwind" fn unwind() {
    panic!();
}

fn main() {
    extern "C" {
        #[unwind(allowed)]
        fn unwind();
    }
    unsafe { unwind() }
}
