// no-system-llvm
// compile-flags: -O -C panic=abort
#![crate_type = "lib"]

fn search<T: Ord + Eq>(arr: &mut [T], a: &T) -> Result<usize, ()> {
    match arr.iter().position(|x| x == a) {
        Some(p) => {
            Ok(p)
        },
        None => Err(()),
    }
}

// CHECK-LABEL: @position_no_bounds_check
#[no_mangle]
pub fn position_no_bounds_check(y: &mut [u32], x: &u32, z: &u32) -> bool {
    // This contains "call assume" so we cannot just rule out all calls
    // CHECK-NOT: panic_bounds_check
    if let Ok(p) = search(y, x) {
      y[p] == *z
    } else {
      false
    }
}

// just to make sure that panicking really emits "panic_bounds_check" somewhere in the IR
// CHECK-LABEL: @test_check
#[no_mangle]
pub fn test_check(y: &[i32]) -> i32 {
    // CHECK: panic_bounds_check
    y[12]
}
