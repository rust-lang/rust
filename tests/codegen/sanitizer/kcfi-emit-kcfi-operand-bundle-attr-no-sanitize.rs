// Verifies that KCFI operand bundles are omitted.
//
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components:
//@ compile-flags: -Cno-prepopulate-passes -Zsanitizer=kcfi -Copt-level=0

#![crate_type="lib"]
#![feature(no_core, no_sanitize, lang_items)]
#![no_core]

#[lang="sized"]
trait Sized { }
#[lang="copy"]
trait Copy { }

impl Copy for i32 {}

#[no_sanitize(kcfi)]
pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: kcfi_emit_kcfi_operand_bundle_attr_no_sanitize::foo
    // CHECK:       Function Attrs: {{.*}}
    // CHECK-LABEL: define{{.*}}foo{{.*}}!{{<unknown kind #36>|kcfi_type}} !{{[0-9]+}}
    // CHECK:       start:
    // CHECK-NOT:   {{%.+}} = call {{(noundef )*}}i32 %f(i32 {{(noundef )*}}%arg){{.*}}[ "kcfi"(i32 {{[-0-9]+}}) ]
    // CHECK:       ret i32 {{%.+}}
    f(arg)
}
