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

#[no_mangle]
pub fn foo() {
    ()
}
