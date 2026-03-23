#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "cdylib"]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "size_of_val"]
trait SizeOfVal: PointeeSized {}
#[lang = "sized"]
trait Sized: SizeOfVal {}

extern "C" {
    fn ir_callee();
}

#[no_mangle]
extern "C" fn rs_foo() {
    unsafe {
        ir_callee();
    }
}
