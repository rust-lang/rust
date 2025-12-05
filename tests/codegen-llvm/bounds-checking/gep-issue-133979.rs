//! Issue: <https://github.com/rust-lang/rust/issues/133979>
//! Check that bounds checking are eliminated.

//@ compile-flags: -Copt-level=2

#![crate_type = "lib"]

// CHECK-LABEL: @test(
#[no_mangle]
fn test(a: &[&[u8]]) -> u32 {
    // CHECK-NOT: panic_bounds_check
    a.iter()
        .enumerate()
        .map(|(y, b)| {
            b.iter()
                .enumerate()
                .filter(|(_, c)| **c == b'A')
                .map(|(x, _)| a[y][x] as u32)
                .sum::<u32>()
        })
        .sum()
}
