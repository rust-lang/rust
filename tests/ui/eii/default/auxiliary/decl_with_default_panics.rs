//@ no-prefer-dynamic
//@ needs-unwind
//@ exec-env:RUST_BACKTRACE=1
#![crate_type = "rlib"]
#![feature(extern_item_impls)]

#[eii(eii1)]
pub fn decl1(x: u64) {
    panic!("{}", x);
}
