// build-pass
// revisions: all strong basic
// compile-flags: --target nvptx64-nvidia-cuda
// needs-llvm-components: nvptx
// [all] compile-flags: -Z stack-protector=all
// [strong] compile-flags: -Z stack-protector=strong
// [basic] compile-flags: -Z stack-protector=basic

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn main(){}
