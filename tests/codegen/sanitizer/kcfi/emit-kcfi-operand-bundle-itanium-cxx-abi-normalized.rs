// Verifies that normalized KCFI type metadata for functions are emitted.
//
//@ add-core-stubs
//@ revisions: aarch64 x86_64
//@ [aarch64] compile-flags: --target aarch64-unknown-none
//@ [aarch64] needs-llvm-components: aarch64
//@ [x86_64] compile-flags: --target x86_64-unknown-none
//@ [x86_64] needs-llvm-components: x86
//@ compile-flags: -Cno-prepopulate-passes -Zsanitizer=kcfi -Zsanitizer-cfi-normalize-integers

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo
    // CHECK-SAME:  {{.*}}!{{<unknown kind #36>|kcfi_type}} ![[TYPE1:[0-9]+]]
    // CHECK:       {{%.+}} = call {{(noundef )*}}i32 %f(i32 {{(noundef )*}}%arg){{.*}}[ "kcfi"(i32 -841055669) ]
    f(arg)
}

pub fn bar(f: fn(i32, i32) -> i32, arg1: i32, arg2: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME:  {{.*}}!{{<unknown kind #36>|kcfi_type}} ![[TYPE2:[0-9]+]]
    // CHECK:       {{%.+}} = call {{(noundef )*}}i32 %f(i32 {{(noundef )*}}%arg1, i32 {{(noundef )*}}%arg2){{.*}}[ "kcfi"(i32 1390819368) ]
    f(arg1, arg2)
}

pub fn baz(f: fn(i32, i32, i32) -> i32, arg1: i32, arg2: i32, arg3: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME:  {{.*}}!{{<unknown kind #36>|kcfi_type}} ![[TYPE3:[0-9]+]]
    // CHECK:       {{%.+}} = call {{(noundef )*}}i32 %f(i32 {{(noundef )*}}%arg1, i32 {{(noundef )*}}%arg2, i32 {{(noundef )*}}%arg3){{.*}}[ "kcfi"(i32 586925835) ]
    f(arg1, arg2, arg3)
}

// CHECK: ![[TYPE1]] = !{i32 -458317079}
// CHECK: ![[TYPE2]] = !{i32 1737138182}
// CHECK: ![[TYPE3]] = !{i32 197182412}
