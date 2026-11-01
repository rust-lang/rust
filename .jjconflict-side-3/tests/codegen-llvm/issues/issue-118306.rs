//@ compile-flags: -Copt-level=3
//@ only-x86_64
//@ revisions: LLVM22 LLVM23
//@ [LLVM22] max-llvm-major-version: 22
//@ [LLVM23] min-llvm-version: 23

// Test for #118306.
// Make sure we don't create `br` or `select` instructions.

#![crate_type = "lib"]

#[no_mangle]
pub fn branchy(input: u64) -> u64 {
    // CHECK-LABEL: @branchy(
    // CHECK-NEXT:  start:
    // CHECK-NEXT:    [[_2:%.*]] = and i64 [[INPUT:%.*]], 3
    // LLVM22-NEXT:   [[SWITCH_GEP:%.*]] = getelementptr inbounds{{( nuw)?}} {{\[4 x i64\]|i64|\[8 x i8\]}}, ptr @switch.table.branchy{{(, i64 0)?}}, i64 [[_2]]
    // LLVM22-NEXT:   [[SWITCH_LOAD:%.*]] = load i64, ptr [[SWITCH_GEP]]
    // LLVM22-NEXT:   ret i64 [[SWITCH_LOAD]]
    // LLVM23-NEXT:   [[SWITCH_GEP:%.*]] = getelementptr inbounds{{( nuw)?}} i8, ptr @switch.table.branchy, i64 [[_2]]
    // LLVM23-NEXT:   [[SWITCH_LOAD:%.*]] = load i8, ptr [[SWITCH_GEP]], align 1
    // LLVM23-NEXT:   [[SWITCH_EXT:%.*]] = zext i8 [[SWITCH_LOAD]] to i64
    // LLVM23-NEXT:   ret i64 [[SWITCH_EXT]]
    match input % 4 {
        1 | 2 => 1,
        3 => 2,
        _ => 0,
    }
}
