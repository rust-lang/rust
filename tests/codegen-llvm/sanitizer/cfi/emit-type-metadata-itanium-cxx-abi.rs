// Verifies that type metadata for functions are emitted.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Ctarget-feature=-crt-static -Zsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo
    // CHECK-SAME:  {{.*}}!type ![[TYPE1:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_E")
    f(arg)
}

pub fn bar(f: fn(i32, i32) -> i32, arg1: i32, arg2: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}bar
    // CHECK-SAME:  {{.*}}!type ![[TYPE2:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_S_E")
    f(arg1, arg2)
}

pub fn baz(f: fn(i32, i32, i32) -> i32, arg1: i32, arg2: i32, arg3: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}baz
    // CHECK-SAME:  {{.*}}!type ![[TYPE3:[0-9]+]] !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"_ZTSFu3i32S_S_S_E")
    f(arg1, arg2, arg3)
}

// CHECK: ![[TYPE1]] = !{i64 0, !"_ZTSFu3i32PFS_S_ES_E"}
// CHECK: ![[TYPE2]] = !{i64 0, !"_ZTSFu3i32PFS_S_S_ES_S_E"}
// CHECK: ![[TYPE3]] = !{i64 0, !"_ZTSFu3i32PFS_S_S_S_ES_S_S_E"}
