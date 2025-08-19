// Verifies that KCFI operand bundles are omitted.
//
//@ add-core-stubs
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Cno-prepopulate-passes -Zsanitizer=kcfi -Copt-level=0

#![crate_type = "lib"]
#![feature(no_core, sanitize, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[sanitize(kcfi = "off")]
pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: emit_kcfi_operand_bundle_attr_sanitize_off::foo
    // CHECK:       Function Attrs: {{.*}}
    // CHECK-LABEL: define{{.*}}foo{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       start:
    // CHECK-NOT:   {{%.+}} = call {{(noundef )*}}i32 %f(i32 {{(noundef )*}}%arg){{.*}}[ "kcfi"(i32 {{[-0-9]+}}) ]
    // CHECK:       ret i32 {{%.+}}
    f(arg)
}
