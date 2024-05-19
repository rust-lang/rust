//@ revisions: rpass1 rpass2 rpass3 rpass4
//@ compile-flags: -Zquery-dep-graph
//@ [rpass1]compile-flags: -Zincremental-ignore-spans
//@ [rpass2]compile-flags: -Zincremental-ignore-spans

#![feature(rustc_attrs)]
#![rustc_partition_reused(module = "change_symbol_export_status-mod1", cfg = "rpass2")]
#![rustc_partition_reused(module = "change_symbol_export_status-mod2", cfg = "rpass2")]
#![rustc_partition_reused(module = "change_symbol_export_status-mod1", cfg = "rpass4")]
#![rustc_partition_reused(module = "change_symbol_export_status-mod2", cfg = "rpass4")]

// This test case makes sure that a change in symbol visibility is detected by
// our dependency tracking. We do this by changing a module's visibility to
// `private` in rpass2, causing the contained function to go from `default` to
// `hidden` visibility.
// The function is marked with #[no_mangle] so it is considered for exporting
// even from an executable. Plain Rust functions are only exported from Rust
// libraries, which our test infrastructure does not support.

#[cfg(any(rpass1,rpass3))]
pub mod mod1 {
    #[no_mangle]
    pub fn foo() {}
}

#[cfg(any(rpass2,rpass4))]
mod mod1 {
    #[no_mangle]
    pub fn foo() {}
}

pub mod mod2 {
    #[no_mangle]
    pub fn bar() {}
}

fn main() {
    mod1::foo();
}
