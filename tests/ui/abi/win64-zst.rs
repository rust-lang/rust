//@ normalize-stderr-test: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
//@ only-x86_64

//@ revisions: x86_64-linux
//@[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86

//@ revisions: x86_64-windows-gnu
//@[x86_64-windows-gnu] compile-flags: --target x86_64-pc-windows-gnu
//@[x86_64-windows-gnu] needs-llvm-components: x86

//@ revisions: x86_64-windows-msvc
//@[x86_64-windows-msvc] compile-flags: --target x86_64-pc-windows-msvc
//@[x86_64-windows-msvc] needs-llvm-components: x86

#![feature(no_core, lang_items, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

#[rustc_abi(debug)]
extern "win64" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
