#![allow(internal_features)]
#![feature(no_core, lang_items, abi_gpu_kernel)]
#![feature(const_trait_impl)]
#![no_core]
#![no_std]

// This is needed because of #![no_core]:
#[lang = "pointee_sized"]
trait PointeeSized {}

#[lang = "meta_sized"]
#[const_trait]
trait MetaSized: PointeeSized {}

#[lang = "sized"]
#[const_trait]
trait Sized: MetaSized {}

#[no_mangle]
extern "gpu-kernel" fn kernel() {}
