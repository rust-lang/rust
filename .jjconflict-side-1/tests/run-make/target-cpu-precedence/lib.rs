#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "rlib"]

#[lang = "pointee_sized"]
#[diagnostic::on_unimplemented(
    message = "values of type `{Self}` may or may not have a size",
    label = "may or may not have a known size"
)]
pub trait PointeeSized {}

#[lang = "meta_sized"]
#[diagnostic::on_unimplemented(
    message = "the size for values of type `{Self}` cannot be known",
    label = "doesn't have a known size"
)]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[diagnostic::on_unimplemented(
    message = "the size for values of type `{Self}` cannot be known at compilation time",
    label = "doesn't have a size known at compile-time"
)]
pub trait Sized: MetaSized {}

// Capture the effective CPU from LLVM IR. This also verifies that the second
// `-Ctarget-cpu` argument took precedence.
// CHECK-LABEL: target triple = "nvptx64-nvidia-cuda"
// CHECK-LABEL: define {{.*}} @foo() {{.*}} #0
// CHECK-LABEL: attributes #0 = {{.*}} "target-cpu"="sm_80" {{.*}}
#[no_mangle]
pub fn foo() {
    ()
}
// The value reconstructed from crate metadata must be identical.
// CHECK-LABEL: =Target modifiers=
// CHECK-LABEL: -Ctarget-cpu=sm_80 [Some("sm_80")]
