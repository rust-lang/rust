// Check the "probe-stack" attribute for targets with `StackProbeType::Call`,
// or `StackProbeType::InlineOrCall` when running on older LLVM.

// compile-flags: -C no-prepopulate-passes
// revisions: i686 x86_64
//[i686] compile-flags: --target i686-unknown-linux-gnu
//[i686] needs-llvm-components: x86
//[i686] ignore-llvm-version: 16 - 99
//[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//[x86_64] needs-llvm-components: x86
//[x86_64] ignore-llvm-version: 16 - 99

#![crate_type = "rlib"]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[no_mangle]
pub fn foo() {
// CHECK: @foo() unnamed_addr #0
// CHECK: attributes #0 = { {{.*}}"probe-stack"="__rust_probestack"{{.*}} }
}
