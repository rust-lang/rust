// Check the "probe-stack" attribute for targets with `StackProbeType::Inline`,
// or `StackProbeType::InlineOrCall` when running on newer LLVM.

// compile-flags: -C no-prepopulate-passes
// revisions: powerpc powerpc64 powerpc64le s390x
//[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//[powerpc] needs-llvm-components: powerpc
//[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//[powerpc64] needs-llvm-components: powerpc
//[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//[powerpc64le] needs-llvm-components: powerpc
//[s390x] compile-flags: --target s390x-unknown-linux-gnu
//[s390x] needs-llvm-components: systemz

#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[no_mangle]
pub fn foo() {
// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"probe-stack"="inline-asm"{{.*}} }
}
