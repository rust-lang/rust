//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn test(a: [i32; 10]) -> i32 {
    // CHECK-LABEL: @test(
    // CHECK: [[L1:%.+]] = load i32
    // CHECK: [[L2:%.+]] = load i32
    // CHECK: [[R:%.+]] = add i32 [[L1]], [[L2]]
    // CHECK: ret i32 [[R]]
    let mut sum = 0;
    for v in a.iter().skip(8) {
        sum += v;
    }

    sum
}
