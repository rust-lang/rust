// compile-flags: -g -O

// Check that each inline call site for the same function uses the same "sub-program" so that LLVM
// can correctly merge the debug info if it merges the inlined code (e.g., for merging of tail
// calls to panic.

// CHECK:       tail call void @_ZN4core9panicking5panic17h{{([0-9a-z]{16})}}E
// CHECK-SAME:  !dbg ![[#first_dbg:]]
// CHECK:       tail call void @_ZN4core9panicking5panic17h{{([0-9a-z]{16})}}E
// CHECK-SAME:  !dbg ![[#second_dbg:]]

// CHECK-DAG:   ![[#func_dbg:]] = distinct !DISubprogram(name: "unwrap<i32>"
// CHECK-DAG:   ![[#first_scope:]] = distinct !DILexicalBlock(scope: ![[#func_dbg]],
// CHECK:       ![[#second_scope:]] = distinct !DILexicalBlock(scope: ![[#func_dbg]],
// CHECK:       ![[#first_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#first_scope]], inlinedAt: ![[#]])
// CHECK:       ![[#second_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#second_scope]], inlinedAt: ![[#]])

#![crate_type = "lib"]

#[no_mangle]
extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
    let x1 = x.unwrap();
    let y1 = y.unwrap();

    x1 + y1
}
