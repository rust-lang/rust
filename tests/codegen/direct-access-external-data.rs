//@ only-loongarch64-unknown-linux-gnu

//@ revisions: default direct indirect
//@ [default] compile-flags: -C relocation-model=static
//@ [direct] compile-flags: -C relocation-model=static -Z direct-access-external-data=yes
//@ [indirect] compile-flags: -C relocation-model=static -Z direct-access-external-data=no

#![crate_type = "rlib"]

// CHECK-DEFAULT: @VAR = external {{.*}} global i32
// CHECK-DIRECT: @VAR = external dso_local {{.*}} global i32
// CHECK-INDIRECT: @VAR = external {{.*}} global i32

extern "C" {
    static VAR: i32;
}

#[no_mangle]
pub fn get() -> i32 {
    unsafe { VAR }
}
