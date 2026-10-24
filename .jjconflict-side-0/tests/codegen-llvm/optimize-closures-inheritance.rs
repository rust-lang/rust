//! Ensure that `#[optimize]` applied to an outer function is inherited by closures
//! and their coercion shims, unless overridden by an explicit `#[optimize]` on the closure.

//@ compile-flags: -C no-prepopulate-passes

#![feature(optimize_attribute)]
#![crate_type = "lib"]

// CHECK-DAG: define{{.*}}test_none{{.*}}call_once{{.*}}#[[ATTR_NONE:[0-9]+]]
// CHECK-DAG: define{{.*}}test_size{{.*}}call_once{{.*}}#[[ATTR_SIZE:[0-9]+]]
// CHECK-DAG: define{{.*}}test_override_size{{.*}}call_once{{.*}}#[[ATTR_SIZE]]
// CHECK-DAG: define{{.*}}test_override_none{{.*}}call_once{{.*}}#[[ATTR_NONE]]

// CHECK-DAG: attributes #[[ATTR_NONE]] = { {{.*}}optnone{{.*}} }
// CHECK-DAG: attributes #[[ATTR_SIZE]] = { {{.*}}optsize{{.*}} }

#[optimize(none)]
#[no_mangle]
pub fn test_none() -> fn() {
    let closure = || ();
    closure
}

#[optimize(size)]
#[no_mangle]
pub fn test_size() -> fn() {
    let closure = || ();
    closure
}

#[optimize(none)]
#[no_mangle]
pub fn test_override_size() -> fn() {
    let closure = {
        #[optimize(size)]
        || ()
    };
    closure
}

#[optimize(size)]
#[no_mangle]
pub fn test_override_none() -> fn() {
    let closure = {
        #[optimize(none)]
        || ()
    };
    closure
}
