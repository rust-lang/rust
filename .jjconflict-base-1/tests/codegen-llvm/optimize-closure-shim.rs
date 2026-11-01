//! Ensure that `#[optimize]` applied to a closure is inherited by its coercion shim.

//@ compile-flags: -Copt-level=3

#![feature(optimize_attribute)]
#![feature(stmt_expr_attributes)]
#![crate_type = "lib"]

// CHECK-DAG: define{{.*}}void {{.*}}shim{{.*}}none{{.*}}(){{.*}}#[[ATTR_NONE:[0-9]+]]
// CHECK-DAG: define{{.*}}void {{.*}}shim{{.*}}size{{.*}}(){{.*}}#[[ATTR_SIZE:[0-9]+]]
// CHECK-DAG: define{{.*}}void {{.*}}shim{{.*}}test_passed_directly{{.*}}(){{.*}}#[[ATTR_SIZE]]

// CHECK-DAG: attributes #[[ATTR_NONE]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes #[[ATTR_SIZE]] = { {{.*}}optsize{{.*}} }

extern "C" {
    fn side_effect_1();
    fn side_effect_2();
    fn side_effect_3();
}

#[no_mangle]
pub fn test_none() -> fn() {
    let closure = #[optimize(none)]
    || unsafe { side_effect_1() };
    closure
}

#[no_mangle]
pub fn test_size() -> fn() {
    let closure = #[optimize(size)]
    || unsafe { side_effect_2() };
    closure
}

#[no_mangle]
#[inline(never)]
pub fn test_passed_directly() {
    f(
        #[optimize(size)]
        || unsafe { side_effect_3() },
    );
}

#[no_mangle]
#[inline(never)]
pub fn f(x: fn()) {
    x();
}
