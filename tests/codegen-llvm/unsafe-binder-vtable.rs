//@ only-64bit
//@ compile-flags: -Znext-solver -Copt-level=3 -Cno-prepopulate-passes

#![crate_type = "lib"]
#![feature(unsafe_binders)]
#![allow(incomplete_features)]

use std::mem::ManuallyDrop;

pub trait Trait<'a> {
    fn method(&self);
}

// Keep the binder in a field: erasing the function signature sees only Wrapped.
// Computing Wrapped's metadata must then follow the field through the binder.
pub struct Wrapped(pub unsafe<'a> ManuallyDrop<dyn Trait<'a> + 'a>);

// The metadata field points to the three vtable header entries and one method.
// CHECK-LABEL: define {{.*}}@binder(
// CHECK-SAME: ptr {{[^,]*}}, ptr {{.*}}dereferenceable(32)
#[no_mangle]
pub fn binder(_: *const Wrapped) {}

// A plain trait object gives the baseline for the binder case above.
// CHECK-LABEL: define {{.*}}@plain(
// CHECK-SAME: ptr {{[^,]*}}, ptr {{.*}}dereferenceable(32)
#[no_mangle]
pub fn plain(_: *const (dyn Trait<'_> + '_)) {}
