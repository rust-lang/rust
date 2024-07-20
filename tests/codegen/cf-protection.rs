// Test that the correct module flags are emitted with different control-flow protection flags.

//@ revisions: undefined none branch return full
//@ needs-llvm-components: x86
//@ [undefined] compile-flags:
//@ [none] compile-flags: -Z cf-protection=none
//@ [branch] compile-flags: -Z cf-protection=branch
//@ [return] compile-flags: -Z cf-protection=return
//@ [full] compile-flags: -Z cf-protection=full
//@ compile-flags: --target x86_64-unknown-linux-gnu

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

// A basic test function.
pub fn test() {}

// CHECK-UNDEFINED-NOT: !"cf-protection-branch"
// CHECK-UNDEFINED-NOT: !"cf-protection-return"

// CHECK-NONE-NOT: !"cf-protection-branch"
// CHECK-NONE-NOT: !"cf-protection-return"

// CHECK-BRANCH-NOT: !"cf-protection-return"
// CHECK-BRANCH: !"cf-protection-branch", i32 1
// CHECK-BRANCH-NOT: !"cf-protection-return"

// CHECK-RETURN-NOT: !"cf-protection-branch"
// CHECK-RETURN: !"cf-protection-return", i32 1
// CHECK-RETURN-NOT: !"cf-protection-branch"

// CHECK-FULL: !"cf-protection-branch", i32 1
// CHECK-FULL: !"cf-protection-return", i32 1
