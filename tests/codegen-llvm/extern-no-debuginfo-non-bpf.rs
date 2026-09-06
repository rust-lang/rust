// Checks that extern declarations do not get debug info outside BPF targets.
//
//@ compile-flags: -C debuginfo=2

#![crate_type = "lib"]

extern "C" {
    // CHECK: @EXTERN_STATIC = external {{.*}}global i32
    // CHECK-NOT: !DIGlobalVariable(name: "EXTERN_STATIC"
    pub static EXTERN_STATIC: i32;
}

extern "C" {
    // CHECK: declare {{.*}}void @extern_fn()
    // CHECK-NOT: !DISubprogram(name: "extern_fn"
    pub fn extern_fn();
}

pub fn use_extern_items() -> i32 {
    unsafe {
        extern_fn();
        EXTERN_STATIC
    }
}
