//@ build-pass
//@ revisions: all strong basic
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx
//@ [all] compile-flags: -Z stack-protector=all
//@ [strong] compile-flags: -Z stack-protector=strong
//@ [basic] compile-flags: -Z stack-protector=basic

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

//[all]~? WARN `-Z stack-protector=all` is not supported for target nvptx64-nvidia-cuda and will be ignored
//[strong]~? WARN `-Z stack-protector=strong` is not supported for target nvptx64-nvidia-cuda and will be ignored
//[basic]~? WARN `-Z stack-protector=basic` is not supported for target nvptx64-nvidia-cuda and will be ignored
