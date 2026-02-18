// Verifies that extern declarations do NOT get debug info on non-BPF targets.
// Debug info for extern declarations is intentionally gated to BPF only.
//
//@ compile-flags: -C debuginfo=2

#![crate_type = "lib"]

extern "C" {
    // CHECK-NOT: !DIGlobalVariable(name: "EXTERN_STATIC"
    pub static EXTERN_STATIC: i32;
}

extern "C" {
    // CHECK-NOT: !DISubprogram(name: "extern_fn"
    pub fn extern_fn();
}

pub fn use_extern_items() -> i32 {
    unsafe {
        extern_fn();
        EXTERN_STATIC
    }
}
