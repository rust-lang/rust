// Verifies that the parent block's debug information are assigned to the inserted cfi block.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cno-prepopulate-passes -Copt-level=0 -Zsanitizer=cfi -Ctarget-feature=-crt-static -Cdebuginfo=1 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo(f: fn(i32) -> i32, arg: i32) -> i32 {
    // CHECK-LABEL: define{{.*}}foo{{.*}}!dbg !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}} !type !{{[0-9]+}}
    // CHECK:       start:
    // CHECK:       [[TT:%.+]] = call i1 @llvm.type.test(ptr {{%f|%0}}, metadata !"{{[[:print:]]+}}"), !dbg !{{[0-9]+}}
    // CHECK-NEXT:  br i1 [[TT]], label %type_test.pass, label %type_test.fail, !dbg !{{[0-9]+}}
    // CHECK:       type_test.pass:                                   ; preds = %start
    // CHECK-NEXT:  {{%.+}} = call i32 %f(i32{{.*}} %arg), !dbg !{{[0-9]+}}
    // CHECK:       type_test.fail:                                   ; preds = %start
    // CHECK-NEXT:  call void @llvm.trap(), !dbg !{{[0-9]+}}
    // CHECK-NEXT:  unreachable, !dbg !{{[0-9]+}}
    f(arg)
}
