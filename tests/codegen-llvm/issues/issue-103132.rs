//@ compile-flags: -Copt-level=3 -C overflow-checks

#![crate_type = "lib"]

#[no_mangle]
pub fn test(arr: &[u8], weight: u32) {
    // CHECK-LABEL: @test(
    // CHECK-NOT: panic
    let weight = weight.min(256 * 256 * 256);

    for x in arr {
        assert!(weight <= 256 * 256 * 256);
        let result = *x as u32 * weight;
    }
}
