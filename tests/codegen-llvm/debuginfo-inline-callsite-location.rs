//@ compile-flags: -g -Copt-level=3 -C panic=abort

// Check that each inline call site for the same function uses the same "sub-program" so that LLVM
// can correctly merge the debug info if it merges the inlined code (e.g., for merging of tail
// calls to panic.

// CHECK:       tail call void @{{[A-Za-z0-9_]+4core6option13unwrap_failed}}
// CHECK-SAME:  !dbg ![[#first_dbg:]]
// CHECK:       tail call void @{{[A-Za-z0-9_]+4core6option13unwrap_failed}}
// CHECK-SAME:  !dbg ![[#second_dbg:]]

// CHECK-DAG:   ![[#func_scope:]] = distinct !DISubprogram(name: "unwrap<i32>"
// CHECK-DAG:   ![[#]] = !DILocalVariable(name: "self",{{( arg: 1,)?}} scope: ![[#func_scope]]
// CHECK:       ![[#first_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#func_scope]], inlinedAt: ![[#]])
// CHECK:       ![[#second_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#func_scope]], inlinedAt: ![[#]])

#![crate_type = "lib"]

#[no_mangle]
extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
    let x1 = x.unwrap();
    let y1 = y.unwrap();

    x1 + y1
}
