#![allow(internal_features)]
#![feature(no_core, lang_items, abi_gpu_kernel)]
#![no_core]
#![no_std]

// This is needed because of #![no_core]:
#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[no_mangle]
extern "gpu-kernel" fn kernel() {}
