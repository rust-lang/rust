// issue: <https://github.com/rust-lang/rust/issues/15562>
// Test resolution of `transmute` in an extern block to rust intrinsics.
//@ run-pass
//@ aux-build:extern-transmute-aux.rs

extern crate extern_transmute_aux as i;

pub fn main() {
    unsafe {
        transmute();
        i::transmute();
    }
}

// We declare this so we don't run into unresolved symbol errors
// The above extern is NOT `extern "rust-intrinsic"` and thus
// means it'll try to find a corresponding symbol to link to.
#[no_mangle]
pub extern "C" fn transmute() {}
