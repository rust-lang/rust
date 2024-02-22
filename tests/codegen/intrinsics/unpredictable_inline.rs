// compile-flags: -O
#![crate_type = "lib"]
#![feature(core_intrinsics)]

#[inline(never)]
#[no_mangle]
pub fn path_a() {
    println!("fast_path");
}

#[inline(never)]
#[no_mangle]
pub fn path_b() {
    println!("slow_path");
}

#[inline(always)]
fn unpredictable(b: bool) -> bool {
    std::intrinsics::unpredictable(b)
}

#[no_mangle]
pub fn f(x: bool) {
    if unpredictable(x) {
        path_a();
    } else {
        path_b();
    }
}

// CHECK-LABEL: @f(
// CHECK: br i1 %x, label %bb1, label %bb2, !unpredictable
// CHECK: bb2:
// CHECK: path_b
// CHECK: bb1:
// CHECK: path_a
