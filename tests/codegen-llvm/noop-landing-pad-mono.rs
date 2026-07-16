//@ compile-flags: -Copt-level=0 -Ccodegen-units=1 -Cno-prepopulate-passes
//@ needs-unwind

// Regression test for <https://github.com/rust-lang/rust/issues/159399>.
//
// `RemoveNoopLandingPads` can't remove the landing pad of `generic_wrapper`
// because it runs on generic MIR, where `T` conservatively needs drop. When
// monomorphizing with a no-drop type like `u32`, codegen must notice that the
// landing pad does nothing and skip it, instead of emitting an `invoke` whose
// landing pad just resumes.

#![crate_type = "lib"]

#[inline(never)]
fn inner(_: &dyn Sync) {}

// CHECK-LABEL: define{{.*}}generic_wrapper
// CHECK-NOT: personality
// CHECK-NOT: invoke
// CHECK-NOT: landingpad
// CHECK-NOT: resume
// CHECK-LABEL: {{^[}]}}
fn generic_wrapper<T: Sync>(val: T) {
    inner(&val);
}

pub fn caller() {
    generic_wrapper(1u32);
}
