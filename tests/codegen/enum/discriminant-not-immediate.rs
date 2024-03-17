// Checks that upon read/write of non-immediate enums that the discriminant has correct layout.
//@ compile-flags: -C no-prepopulate-passes
#![crate_type = "lib"]

// CHECK-LABEL: @write_option_to_memory(
#[no_mangle]
pub fn write_option_to_memory(store: &mut Option<u32>, reg: Option<u32>) {
    // CHECK: %[[Z:.+]] = zext i1 %reg.0 to i32
    // CHECK: store i32 %[[Z]], ptr %store, align 4
    // CHECK: %[[P:.+]] = getelementptr inbounds i8, ptr %store, i64 4
    // CHECK: store i32 %reg.1, ptr %[[P]], align 4
    *store = reg;
}

// CHECK-LABEL: @read_option_from_memory(
#[no_mangle]
pub fn read_option_from_memory(store: &Option<u32>) -> Option<u32> {
    // CHECK: %[[RAW_STORE_DISCRIM:.+]] = load i32, ptr %store, align 4
    // CHECK-SAME: !range
    // CHECK-SAME: !noundef
    // CHECK: %[[RET:.+]].0 = trunc i32 %[[RAW_STORE_DISCRIM]] to i1
    // CHECK: %[[RAW_STORE_T:.+]] = getelementptr inbounds i8, ptr %store, i64 4
    // CHECK: %[[RET]].1 = load i32, ptr %[[RAW_STORE_T]], align 4
    // CHECK: %[[IMM:.+]] = insertvalue { i1, i32 } poison, i1 %[[RET]].0, 0
    // CHECK: %[[R:.+]] = insertvalue { i1, i32 } %[[IMM]], i32 %[[RET]].1, 1
    // CHECK: ret { i1, i32 } %[[R]]
    *store
}
