//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

#[inline(never)]
#[no_mangle]
pub fn takes_pointer(_: *const u8) {}

// CHECK-LABEL: @has_local_zsts
#[no_mangle]
pub fn has_local_zsts() {
    // CHECK-NOT: alloca

    // CHECK: call void @takes_pointer(ptr getelementptr (i8, ptr null, {{i64 -9223372036854775808|i32 -2147483648|i16 -32768}}))
    let unit = ();
    takes_pointer((&raw const unit) as _);

    // CHECK: call void @takes_pointer(ptr getelementptr (i8, ptr null, {{i64 -9223372036854775808|i32 -2147483648|i16 -32768}}))
    let array = [0_u64; 0];
    takes_pointer((&raw const array) as _);
}
