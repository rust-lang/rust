//@ only-loongarch64-unknown-linux-gnu

//@ revisions: DEFAULT DIRECT INDIRECT
//@ [DEFAULT] compile-flags: -C relocation-model=static
//@ [DIRECT] compile-flags: -C relocation-model=static -Z direct-access-external-data=yes
//@ [INDIRECT] compile-flags: -C relocation-model=static -Z direct-access-external-data=no

#![crate_type = "rlib"]

// DEFAULT: @VAR = external {{.*}} global i32
// DIRECT: @VAR = external dso_local {{.*}} global i32
// INDIRECT: @VAR = external {{.*}} global i32

extern "C" {
    static VAR: i32;
}

#[no_mangle]
pub fn get() -> i32 {
    unsafe { VAR }
}
