#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

// This is needed because of #![no_core]:
#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "size_of_val"]
trait SizeOfVal: PointeeSized {}
#[lang = "sized"]
trait Sized: SizeOfVal {}

#[link(name = "extern_1", kind = "raw-dylib")]
extern "C" {
    fn extern_fn();
}

pub fn extern_fn_caller() {
    unsafe {
        extern_fn();
    }
}
