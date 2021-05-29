// error-pattern: calling a function with ABI C-unwind using caller ABI C
#![feature(c_unwind)]

//! Unwinding when the caller ABI is "C" (without "-unwind") is UB.
//! Currently we detect the ABI mismatch; we could probably allow such calls in principle one day
//! but then we have to detect the unexpected unwinding.
//! FIXME: `-Zmiri-disable-abi-check` does not work for this test because function pointers are
//! always allowed to unwind.

extern "C-unwind" fn unwind() {
    panic!();
}

fn main() {
    let unwind: extern "C-unwind" fn() = unwind;
    let unwind: extern "C" fn() = unsafe { std::mem::transmute(unwind) };
    std::panic::catch_unwind(|| unwind()).unwrap_err();
}
