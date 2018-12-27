// run-pass
// aux-build:issue-15562.rs

// pretty-expanded FIXME #23616

extern crate issue_15562 as i;

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
pub extern fn transmute() {}
