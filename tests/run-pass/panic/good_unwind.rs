#![feature(c_unwind, unboxed_closures, unwind_attributes)]

use std::panic;

extern "C-unwind" fn good_unwind_c() {
    panic!();
}

#[unwind(allowed)]
extern "C" fn good_unwind_allowed() {
    panic!();
}

fn good_unwind_rust() {
    panic!();
}

extern "rust-call" fn good_unwind_rust_call(_: ()) {
    panic!();
}

fn main() {
    panic::catch_unwind(|| good_unwind_c()).unwrap_err();
    panic::catch_unwind(|| good_unwind_allowed()).unwrap_err();
    panic::catch_unwind(|| good_unwind_rust()).unwrap_err();
    good_unwind_rust_call(());
}
