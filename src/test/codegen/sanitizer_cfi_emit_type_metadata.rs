// Verifies that type metadata for functions are emitted.
//
// ignore-windows
// needs-sanitizer-cfi
// only-aarch64
// only-x86_64
// compile-flags: -Clto -Cno-prepopulate-passes -Zsanitizer=cfi

#![crate_type="lib"]

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo{{.*}}!type !{{[0-9]+}}
    // CHECK:       %1 = call i1 @llvm.type.test(i8* %0, metadata !"typeid1")
    f(arg)
}

pub fn bar(f: fn(i32, i32) -> i32, arg1: i32, arg2: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}bar{{.*}}!type !{{[0-9]+}}
    // CHECK:       %1 = call i1 @llvm.type.test(i8* %0, metadata !"typeid2")
    f(arg1, arg2)
}

pub fn baz(f: fn(i32, i32, i32) -> i32, arg1: i32, arg2: i32, arg3: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}baz{{.*}}!type !{{[0-9]+}}
    // CHECK:       %1 = call i1 @llvm.type.test(i8* %0, metadata !"typeid3")
    f(arg1, arg2, arg3)
}

// CHECK: !{{[0-9]+}} = !{i64 0, !"typeid2"}
// CHECK: !{{[0-9]+}} = !{i64 0, !"typeid3"}
// CHECK: !{{[0-9]+}} = !{i64 0, !"typeid4"}
