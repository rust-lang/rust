//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Cpanic=abort
// Check that there's an alloca for the reference and the vector, but nothing else.
// We use panic=abort because unwinding panics give hint::black_box a cleanup block, which has
// another alloca.

#![crate_type = "lib"]

#[inline(never)]
fn test<const SIZE: usize>() {
    // CHECK-LABEL: no_alloca_inside_if_false::test
    // CHECK: start:
    // CHECK-NEXT: alloca [{{12|24}} x i8]
    // CHECK-NOT: alloca
    if const { SIZE < 4096 } {
        let arr = [0u8; SIZE];
        std::hint::black_box(&arr);
    } else {
        let vec = vec![0u8; SIZE];
        std::hint::black_box(&vec);
    }
}

// CHECK-LABEL: @main
#[no_mangle]
pub fn main() {
    test::<8192>();
}
