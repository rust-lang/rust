// Verifies that pointer type membership tests for indirect calls are omitted.
//
// needs-sanitizer-cfi
// compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0

#![crate_type="lib"]
#![feature(no_sanitize)]

#[no_sanitize(cfi)]
pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: cfi_emit_type_checks_attr_no_sanitize::foo
    // CHECK:       Function Attrs: {{.*}}
    // CHECK-LABEL: define{{.*}}foo{{.*}}!type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       start:
    // CHECK-NEXT:  {{%.+}} = call i32 %f(i32 %arg)
    // CHECK-NEXT:  ret i32 {{%.+}}
    f(arg)
}
