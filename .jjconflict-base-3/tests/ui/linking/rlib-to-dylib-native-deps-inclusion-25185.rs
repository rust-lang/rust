// https://github.com/rust-lang/rust/issues/25185
//@ run-pass
//@ aux-build:aux-25185-1.rs
//@ aux-build:aux-25185-2.rs

extern crate aux_25185_2;

fn main() {
    let x = unsafe {
        aux_25185_2::rust_dbg_extern_identity_u32(1)
    };
    assert_eq!(x, 1);
}
