// Verifies that pointer type membership tests for indirect calls are emitted.
//
// ignore-windows
// needs-sanitizer-cfi
// only-aarch64
// only-x86_64
// compile-flags: -Clto -Cno-prepopulate-passes -Zsanitizer=cfi

#![crate_type="lib"]

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo{{.*}}!type !{{[0-9]+}}
    // CHECK:       start:
    // CHECK-NEXT:  %0 = bitcast i32 (i32)* %f to i8*
    // CHECK-NEXT:  %1 = call i1 @llvm.type.test(i8* %0, metadata !"{{[[:print:]]+}}")
    // CHECK-NEXT:  br i1 %1, label %type_test.pass, label %type_test.fail
    // CHECK:       type_test.pass:
    // CHECK-NEXT:  %2 = call i32 %f(i32 %arg)
    // CHECK-NEXT:  br label %bb1
    // CHECK:       type_test.fail:
    // CHECK-NEXT:  call void @llvm.trap()
    // CHECK-NEXT:  unreachable
    f(arg)
}
