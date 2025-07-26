//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn test() -> u32 {
    // CHECK-LABEL: @test(
    // CHECK: ret i32 13
    let s = [1, 2, 3, 4, 5, 6, 7];

    let mut iter = s.iter();
    let mut sum = 0;
    while let Some(_) = iter.next() {
        sum += iter.next().map_or(1, |&x| x)
    }

    sum
}
