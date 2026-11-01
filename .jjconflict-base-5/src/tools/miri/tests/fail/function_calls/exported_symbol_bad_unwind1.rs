#[no_mangle]
extern "C-unwind" fn unwind() {
    panic!();
}

fn main() {
    extern "C" {
        fn unwind();
    }
    unsafe { unwind() }
    //~^ ERROR: unwinding past a stack frame that does not allow unwinding
}
