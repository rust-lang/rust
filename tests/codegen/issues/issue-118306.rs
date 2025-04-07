//@ compile-flags: -Copt-level=3
//@ only-x86_64

// Test for #118306.
// Make sure we don't create `br` or `select` instructions.

#![crate_type = "lib"]

#[no_mangle]
pub fn branchy(input: u64) -> u64 {
    // CHECK-LABEL: @branchy(
    // CHECK-NEXT:  start:
    // CHECK-NEXT:    [[_2:%.*]] = and i64 [[INPUT:%.*]], 3
    // CHECK-NEXT:    [[SWITCH_GEP:%.*]] = getelementptr inbounds{{( nuw)?}} [4 x i64], ptr @switch.table.branchy, i64 0, i64 [[_2]]
    // CHECK-NEXT:    [[SWITCH_LOAD:%.*]] = load i64, ptr [[SWITCH_GEP]]
    // CHECK-NEXT:    ret i64 [[SWITCH_LOAD]]
    match input % 4 {
        1 | 2 => 1,
        3 => 2,
        _ => 0,
    }
}
