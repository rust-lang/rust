//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0

#![crate_type = "lib"]

#[inline(never)]
fn test<const SIZE: usize>() {
    // CHECK-LABEL: no_alloca_inside_if_false::test
    // CHECK: start:
    // CHECK-NEXT: %0 = alloca
    // CHECK-NEXT: %vec = alloca
    // CHECK-NOT: %arr = alloca
    if const { SIZE < 4096 } {
        let arr = [0u8; SIZE];
        std::hint::black_box(&arr);
    } else {
        let vec = vec![0u8; SIZE];
        std::hint::black_box(&vec);
    }
}

pub fn main() {
    test::<8192>();
}
