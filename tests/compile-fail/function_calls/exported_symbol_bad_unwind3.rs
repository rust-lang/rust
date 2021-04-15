#![feature(unwind_attributes)]
#![feature(c_unwind)] // make sure it doesn't insert abort-on-unwind for the `#[unwind(allowed)]` function

#[unwind(allowed)]
#[no_mangle]
extern "C" fn unwind() {
    panic!();
}

fn main() {
    extern "C" {
        fn unwind();
    }
    unsafe { unwind() }
    //~^ ERROR unwinding past a stack frame that does not allow unwinding
}
