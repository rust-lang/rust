// Verifies that normalized and generalized type metadata for functions are emitted.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers -Zsanitizer-cfi-generalize-pointers -C unsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-normalize-integers

#![crate_type = "lib"]

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo
    // CHECK-SAME:  {{.*}}![[TYPE1:[0-9]+]]
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_E.normalized.generalized")
    f(arg)
}

pub fn bar(f: fn(i32, i32) -> i32, arg1: i32, arg2: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME:  {{.*}}![[TYPE2:[0-9]+]]
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_S_E.normalized.generalized")
    f(arg1, arg2)
}

pub fn baz(f: fn(i32, i32, i32) -> i32, arg1: i32, arg2: i32, arg3: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME:  {{.*}}![[TYPE3:[0-9]+]]
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_S_S_E.normalized.generalized")
    f(arg1, arg2, arg3)
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFu3i32PKvS_E.normalized.generalized"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFu3i32PKvS_S_E.normalized.generalized"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFu3i32PKvS_S_S_E.normalized.generalized"}
