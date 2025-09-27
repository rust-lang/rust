//@ check-fail
//@ revisions: all strong all-z
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx
//@ [all] compile-flags: -C stack-protector=all
//@ [strong] compile-flags: -C stack-protector=strong
//@ [all-z] compile-flags: -Z stack-protector=all

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
trait Sized: MetaSized {}

#[lang = "copy"]
trait Copy {}

pub fn main(){}

//[all]~? ERROR `-C stack-protector=all` is not supported for target nvptx64-nvidia-cuda
//[all-z]~? ERROR `-C stack-protector=all` is not supported for target nvptx64-nvidia-cuda
//[strong]~? ERROR `-C stack-protector=strong` is not supported for target nvptx64-nvidia-cuda
