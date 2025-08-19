#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "cdylib"]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}

extern "C" {
    fn ir_callee();
}

#[no_mangle]
extern "C" fn rs_foo() {
    unsafe {
        ir_callee();
    }
}
