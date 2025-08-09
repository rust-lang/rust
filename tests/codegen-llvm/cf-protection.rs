// Test that the correct module flags are emitted with different control-flow protection flags.

//@ add-core-stubs
//@ revisions: undefined none branch return full
//@ needs-llvm-components: x86
// [undefined] no extra compile-flags
//@ [none] compile-flags: -Z cf-protection=none
//@ [branch] compile-flags: -Z cf-protection=branch
//@ [return] compile-flags: -Z cf-protection=return
//@ [full] compile-flags: -Z cf-protection=full
//@ compile-flags: --target x86_64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

// A basic test function.
pub fn test() {}

// undefined-NOT: !"cf-protection-branch"
// undefined-NOT: !"cf-protection-return"

// none-NOT: !"cf-protection-branch"
// none-NOT: !"cf-protection-return"

// branch-NOT: !"cf-protection-return"
// branch: !"cf-protection-branch", i32 1
// branch-NOT: !"cf-protection-return"

// return-NOT: !"cf-protection-branch"
// return: !"cf-protection-return", i32 1
// return-NOT: !"cf-protection-branch"

// full: !"cf-protection-branch", i32 1
// full: !"cf-protection-return", i32 1
