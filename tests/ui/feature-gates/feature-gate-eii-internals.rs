#![crate_type = "rlib"]
#![feature(decl_macro)]
#![feature(rustc_attrs)]

#[eii_macro_for(bar)] //~ ERROR use of unstable library feature `eii_internals`
#[rustc_builtin_macro(eii_macro)]
macro foo() {

}

unsafe extern "Rust" {
    safe fn bar(x: u64) -> u64;
}
