//! Ensure that `#[optimize]` applied to an outer function is inherited by closures
//! and their coercion shims, unless overridden by an explicit `#[optimize]` on the closure.

//@ compile-flags: -Copt-level=3

#![feature(optimize_attribute)]
#![crate_type = "lib"]

// CHECK-DAG: define{{.*}}void {{.*}}test_none{{.*}}call_once{{.*}}#[[ATTR_NONE:[0-9]+]]
// CHECK-DAG: define{{.*}}void {{.*}}test_none{{.*}}B{{[0-9]+}}_() {{.*}}#[[ATTR_NONE]]
// CHECK-DAG: define{{.*}}void {{.*}}test_size{{.*}}call_once{{.*}}#[[ATTR_SIZE:[0-9]+]]
// CHECK-DAG: define{{.*}}void {{.*}}test_override_size{{.*}}call_once{{.*}}#[[ATTR_SIZE]]
// CHECK-DAG: define{{.*}}void {{.*}}test_override_none{{.*}}call_once{{.*}}#[[ATTR_NONE]]

// CHECK-DAG: attributes #[[ATTR_NONE]] = { {{.*}}noinline{{.*}}optnone{{.*}} }
// CHECK-DAG: attributes #[[ATTR_SIZE]] = { {{.*}}optsize{{.*}} }

extern "C" {
    fn side_effect_1();
    fn side_effect_2();
    fn side_effect_3();
    fn side_effect_4();
}

#[optimize(none)]
#[no_mangle]
pub fn test_none() -> fn() {
    let closure = || unsafe { side_effect_1() };
    closure
}

#[optimize(size)]
#[no_mangle]
pub fn test_size() -> fn() {
    let closure = || unsafe { side_effect_2() };
    closure
}

#[optimize(none)]
#[no_mangle]
pub fn test_override_size() -> fn() {
    let closure = {
        #[optimize(size)]
        || unsafe { side_effect_3() }
    };
    closure
}

#[optimize(size)]
#[no_mangle]
pub fn test_override_none() -> fn() {
    let closure = {
        #[optimize(none)]
        || unsafe { side_effect_4() }
    };
    closure
}
