// Verifies that #[link_section] works on foreign (extern) items.
//
//@ ignore-wasm32 custom sections work differently on wasm
//@ compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

extern "C" {
    // CHECK: @EXTERN_STATIC = external global i32, section ".ksyms"
    #[link_section = ".ksyms"]
    pub static EXTERN_STATIC: i32;
}

extern "C" {
    // CHECK: declare {{.*}}void @extern_fn(){{.*}} section ".ksyms"
    #[link_section = ".ksyms"]
    pub fn extern_fn();
}

// Ensure the extern items are used so they appear in the IR
pub fn use_extern_items() -> i32 {
    unsafe {
        extern_fn();
        EXTERN_STATIC
    }
}
