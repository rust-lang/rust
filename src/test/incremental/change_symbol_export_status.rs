// revisions: rpass1 rpass2
// compile-flags: -Zquery-dep-graph

#![feature(rustc_attrs)]
#![allow(private_no_mangle_fns)]

#![rustc_partition_codegened(module="change_symbol_export_status-mod1", cfg="rpass2")]
#![rustc_partition_reused(module="change_symbol_export_status-mod2", cfg="rpass2")]

// This test case makes sure that a change in symbol visibility is detected by
// our dependency tracking. We do this by changing a module's visibility to
// `private` in rpass2, causing the contained function to go from `default` to
// `hidden` visibility.
// The function is marked with #[no_mangle] so it is considered for exporting
// even from an executable. Plain Rust functions are only exported from Rust
// libraries, which our test infrastructure does not support.

#[cfg(rpass1)]
pub mod mod1 {
    #[no_mangle]
    pub fn foo() {}
}

#[cfg(rpass2)]
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
