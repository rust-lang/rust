//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn test(a: i32, b: i32) -> bool {
    // CHECK-LABEL: @test(
    // CHECK: ret i1 true
    let c1 = (a >= 0) && (a <= 10);
    let c2 = (b >= 0) && (b <= 20);

    if c1 & c2 { a + 100 != b } else { true }
}
