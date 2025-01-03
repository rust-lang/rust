//@ compile-flags: -O

#![feature(select_unpredictable)]
#![crate_type = "lib"]

#[no_mangle]
pub fn test_int(p: bool, a: u64, b: u64) -> u64 {
    // CHECK-LABEL: define{{.*}} @test_int
    // CHECK: select i1 %p, i64 %a, i64 %b, !unpredictable
    p.select_unpredictable(a, b)
}

#[no_mangle]
pub fn test_pair(p: bool, a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
    // CHECK-LABEL: define{{.*}} @test_pair
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    p.select_unpredictable(a, b)
}

struct Large {
    e: [u64; 100],
}

#[no_mangle]
pub fn test_struct(p: bool, a: Large, b: Large) -> Large {
    // CHECK-LABEL: define{{.*}} @test_struct
    // CHECK: select i1 %p, {{.*}}, !unpredictable
    p.select_unpredictable(a, b)
}

#[no_mangle]
pub fn test_zst(p: bool, a: (), b: ()) -> () {
    // CHECK-LABEL: define{{.*}} @test_zst
    p.select_unpredictable(a, b)
}
