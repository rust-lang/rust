// Verifies that pointer type membership tests for indirect calls are omitted.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(sanitize)]

#[sanitize(cfi = "off")]
pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: emit_type_checks_attr_sanitize_off::foo
    // CHECK:       Function Attrs: {{.*}}
    // CHECK-LABEL: define{{.*}}foo{{.*}}!type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       start:
    // CHECK-NEXT:  {{%.+}} = call i32 %f(i32{{.*}} %arg)
    // CHECK-NEXT:  ret i32 {{%.+}}
    f(arg)
}
