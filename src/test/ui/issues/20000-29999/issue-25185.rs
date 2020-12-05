// run-pass
// aux-build:issue-25185-1.rs
// aux-build:issue-25185-2.rs
// ignore-wasm32-bare no libc for ffi testing

extern crate issue_25185_2;

fn main() {
    let x = unsafe {
        issue_25185_2::rust_dbg_extern_identity_u32(1)
    };
    assert_eq!(x, 1);
}
