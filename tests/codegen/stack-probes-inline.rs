// Check the "probe-stack" attribute for targets with `StackProbeType::Inline`,
// or `StackProbeType::InlineOrCall` when running on newer LLVM.

//@ add-core-stubs
//@ compile-flags: -C no-prepopulate-passes
//@ revisions: aarch64 powerpc powerpc64 powerpc64le s390x i686 x86_64
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//@[powerpc] needs-llvm-components: powerpc
//@[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//@[powerpc64] needs-llvm-components: powerpc
//@[powerpc64le] compile-flags: --target powerpc64le-unknown-linux-gnu
//@[powerpc64le] needs-llvm-components: powerpc
//@[s390x] compile-flags: --target s390x-unknown-linux-gnu
//@[s390x] needs-llvm-components: systemz
//@[i686] compile-flags: --target i686-unknown-linux-gnu
//@[i686] needs-llvm-components: x86
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86

#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
pub fn foo() {
    // CHECK: @foo() unnamed_addr #0
    // CHECK: attributes #0 = { {{.*}}"probe-stack"="inline-asm"{{.*}} }
}
