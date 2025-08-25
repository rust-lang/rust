//@ no-prefer-dynamic
#![crate_type = "rlib"]
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_extern_target(bar)]
#[rustc_builtin_macro(eii_macro)]
pub macro foo() {}

unsafe extern "Rust" {
    pub safe fn bar(x: u64) -> u64;
}
