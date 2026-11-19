// Always setting -Zmir-opt-level to overwrite the test harness.
// Assumes that -Copt-level=0 maps to mir-opt-level=1, and other -Copt-level map to
// mir-opt-level=2.
//@ revisions: NO-OPT SPEED-OPT ABSOLUTELY-NO-OPT
//@[NO-OPT] compile-flags: -Copt-level=0 -Zmir-opt-level=1
//@[ABSOLUTELY-NO-OPT] compile-flags: -Copt-level=0 -Zmir-opt-level=0
//@[SPEED-OPT] compile-flags: -Copt-level=3 -Zmir-opt-level=2 -Coverflow-checks=y
#![feature(core_intrinsics)]

#[optimize(none)]
pub fn add_noopt() -> i32 {
    // CHECK-LABEL: fn add_noopt(
    // CHECK: AddWithOverflow(const 1_i32, const 2_i32);
    // CHECK-NEXT: assert
    1 + 2
}

#[optimize(none)]
#[allow(unused_assignments)]
pub fn dead_store_noopt(input: i32) -> i32 {
    // CHECK-LABEL: fn dead_store_noopt(
    // CHECK: debug value => [[VALUE:_[0-9]+]];
    // CHECK: [[VALUE]] = copy _1;
    // CHECK-NEXT: [[VALUE]] = const 1_i32;
    // CHECK-NEXT: [[VALUE]] = const 2_i32;
    // CHECK-NEXT: _0 = copy [[VALUE]];

    let mut value = input;
    value = 1;
    value = 2;
    value
}

// Ensure that with mir-opt-level 0, we do not run InstSimplify
#[optimize(none)]
pub fn align_of_val_noopt() -> usize {
    // CHECK-LABEL: fn align_of_val_noopt(
    unsafe {
        let ptr: *const [u8] = std::mem::transmute((42usize, usize::MAX));
        // InstCombine would optimize away the intrinsic call here.
        // ABSOLUTELY-NO-OPT: intrinsics::align_of_val
        // With mir-opt-level 1, we do run InstSimplify but don't compute the final constant.
        // NO-OPT: const <u8 as std::mem::SizedTypeProperties>::ALIGN
        // SPEED-OPT: const <u8 as std::mem::SizedTypeProperties>::ALIGN
        std::intrinsics::align_of_val(ptr)
    }
}

// Even with optimize(speed), a mir-opt-level of 0 suppresses MIR opts.
#[optimize(speed)]
pub fn align_of_val_opt() -> usize {
    // CHECK-LABEL: fn align_of_val_opt(
    unsafe {
        let ptr: *const [u8] = std::mem::transmute((42usize, usize::MAX));
        // ABSOLUTELY-NO-OPT: intrinsics::align_of_val
        // NO-OPT: _0 = const <u8 as std::mem::SizedTypeProperties>::ALIGN;
        // SPEED-OPT: _0 = const 1_usize;
        std::intrinsics::align_of_val(ptr)
    }
}

fn main() {}
