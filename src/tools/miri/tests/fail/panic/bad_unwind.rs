//! Unwinding when the caller ABI is "C" (without "-unwind") is UB.
// The opposite version (callee does not allow unwinding) is impossible to
// even write: MIR validation catches functions that have `UnwindContinue` but
// are not allowed to unwind.

extern "C-unwind" fn unwind() {
    panic!();
}

fn main() {
    let unwind: extern "C-unwind" fn() = unwind;
    let unwind: extern "C" fn() = unsafe { std::mem::transmute(unwind) };
    std::panic::catch_unwind(|| unwind()).unwrap_err();
    //~^ ERROR: unwinding past a stack frame that does not allow unwinding
}
