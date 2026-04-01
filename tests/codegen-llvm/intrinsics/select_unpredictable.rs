//@ compile-flags: -Copt-level=3 -Zmerge-functions=disabled

#![feature(core_intrinsics)]
#![crate_type = "lib"]

/* Test the intrinsic */

#[no_mangle]
pub fn test_int(p: bool, a: u64, b: u64) -> u64 {
    // CHECK-LABEL: define{{.*}} @test_int
    // CHECK: select i1 %p, i64 %a, i64 %b, !unpredictable
    core::intrinsics::select_unpredictable(p, a, b)
}

#[no_mangle]
pub fn test_pair(p: bool, a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    // CHECK-LABEL: define{{.*}} @test_pair
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    core::intrinsics::select_unpredictable(p, a, b)
}

struct Large {
    e: [u64; 100],
}

#[no_mangle]
pub fn test_struct(p: bool, a: Large, b: Large) -> Large {
    // CHECK-LABEL: define{{.*}} @test_struct
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    core::intrinsics::select_unpredictable(p, a, b)
}

// ZSTs should not need a `select` expression.
#[no_mangle]
pub fn test_zst(p: bool, a: (), b: ()) -> () {
    // CHECK-LABEL: define{{.*}} @test_zst
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    core::intrinsics::select_unpredictable(p, a, b)
}

/* Test the user-facing version */

#[no_mangle]
pub fn test_int2(p: bool, a: u64, b: u64) -> u64 {
    // CHECK-LABEL: define{{.*}} @test_int2
    // CHECK: select i1 %p, i64 %a, i64 %b, !unpredictable
    core::hint::select_unpredictable(p, a, b)
}

#[no_mangle]
pub fn test_pair2(p: bool, a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    // CHECK-LABEL: define{{.*}} @test_pair2
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    core::hint::select_unpredictable(p, a, b)
}

#[no_mangle]
pub fn test_struct2(p: bool, a: Large, b: Large) -> Large {
    // CHECK-LABEL: define{{.*}} @test_struct2
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    core::hint::select_unpredictable(p, a, b)
}

#[no_mangle]
pub fn test_zst2(p: bool, a: (), b: ()) -> () {
    // CHECK-LABEL: define{{.*}} @test_zst2
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void
    core::hint::select_unpredictable(p, a, b)
}
