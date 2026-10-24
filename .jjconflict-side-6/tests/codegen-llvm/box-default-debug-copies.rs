//@ compile-flags: -Copt-level=0

// Test to make sure that `<Box<T>>::default` does not create too many copies of `T` on the stack.
// in debug mode. This regressed in dd0620b86721ae8cae86736443acd3f72ba6fc32 to
// four `T` allocas.
//
// See https://github.com/rust-lang/rust/issues/136043 for more context.
//
// FIXME: This test only wants to ensure that there are at most two allocas of `T` created, instead
// of checking for exactly two.

#![crate_type = "lib"]

#[allow(dead_code)]
pub struct Thing([u8; 1000000]);

impl Default for Thing {
    fn default() -> Self {
        Thing([0; 1000000])
    }
}

// CHECK-COUNT-2: %{{.*}} = alloca {{.*}}1000000
// CHECK-NOT: %{{.*}} = alloca {{.*}}1000000
#[no_mangle]
pub fn box_default_single_copy() -> Box<Thing> {
    Box::default()
}
