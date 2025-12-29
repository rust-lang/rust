//@ ignore-powerpc64 (handles dso_local differently)
//@ ignore-apple (handles dso_local differently)

//@ revisions: DEFAULT PIE DIRECT INDIRECT
//@ [DEFAULT] compile-flags: -C relocation-model=static
//@ [PIE] compile-flags: -C relocation-model=pie
//@ [DIRECT] compile-flags: -C relocation-model=pie -Z direct-access-external-data=yes
//@ [INDIRECT] compile-flags: -C relocation-model=static -Z direct-access-external-data=no

#![crate_type = "rlib"]

unsafe extern "C" {
    // CHECK: @VAR = external
    // DEFAULT-SAME: dso_local
    // PIE-NOT: dso_local
    // DIRECT-SAME: dso_local
    // INDIRECT-NOT: dso_local
    // CHECK-SAME: global i32
    safe static VAR: i32;
}

#[no_mangle]
pub fn refer() {
    core::hint::black_box(VAR);
}
