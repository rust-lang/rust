//@ revisions: all strong strong-ok basic basic-ok
//@ compile-flags: --target x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86
//@ [all] check-pass
//@ [all] compile-flags: -C stack-protector=all
//@ [strong] check-fail
//@ [strong] compile-flags: -C stack-protector=strong
//@ [strong-ok] check-pass
//@ [strong-ok] compile-flags: -C stack-protector=strong -Z unstable-options
//@ [basic] check-fail
//@ [basic] compile-flags: -C stack-protector=basic
//@ [basic-ok] check-pass
//@ [basic-ok] compile-flags: -C stack-protector=basic -Z unstable-options

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

pub fn main(){}
