//@ ignore-powerpc64 (handles dso_local differently)
//@ ignore-apple (handles dso_local differently)

//@ revisions: DEFAULT PIE DIRECT INDIRECT
//@ [DEFAULT] compile-flags: -C relocation-model=static
//@ [PIE] compile-flags: -C relocation-model=pie
//@ [DIRECT] compile-flags: -C relocation-model=pie -Z direct-access-external-data=yes
//@ [INDIRECT] compile-flags: -C relocation-model=static -Z direct-access-external-data=no

#![crate_type = "rlib"]
#![feature(linkage)]

unsafe extern "C" {
    // CHECK: @VAR = external
    // DEFAULT-SAME: dso_local
    // PIE-NOT: dso_local
    // DIRECT-SAME: dso_local
    // INDIRECT-NOT: dso_local
    // CHECK-SAME: global i32
    safe static VAR: i32;

    // When "linkage" is used, we generate an indirection global.
    // Check dso_local is still applied to the actual global.
    // CHECK: @EXTERNAL = external
    // DEFAULT-SAME: dso_local
    // PIE-NOT: dso_local
    // DIRECT-SAME: dso_local
    // INDIRECT-NOT: dso_local
    // CHECK-SAME: global i8
    #[linkage = "external"]
    safe static EXTERNAL: *const u32;

    // CHECK: @WEAK = extern_weak
    // DEFAULT-SAME: dso_local
    // PIE-NOT: dso_local
    // DIRECT-SAME: dso_local
    // INDIRECT-NOT: dso_local
    // CHECK-SAME: global i8
    #[linkage = "extern_weak"]
    safe static WEAK: *const u32;
}

#[no_mangle]
pub fn refer() {
    core::hint::black_box(VAR);
    core::hint::black_box(EXTERNAL);
    core::hint::black_box(WEAK);
}
