#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "cdylib"]

#[lang = "sized"]
trait Sized {}

extern "C" {
    fn ir_callee();
}

#[no_mangle]
extern "C" fn rs_foo() {
    unsafe {
        ir_callee();
    }
}
