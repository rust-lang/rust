// compile-flags: -g -O

// Check that each inline call site for the same function uses the same "sub-program" so that LLVM
// can correctly merge the debug info if it merges the inlined code (e.g., for merging of tail
// calls to panic.

// Handle both legacy and v0 symbol mangling.
// CHECK:       tail call void @{{.*core9panicking5panic}}
// CHECK-SAME:  !dbg ![[#first_dbg:]]
// CHECK:       tail call void @{{.*core9panicking5panic}}
// CHECK-SAME:  !dbg ![[#second_dbg:]]

// CHECK:       ![[#func_dbg:]] = distinct !DISubprogram(name: "unwrap<i32>"
// CHECK:       ![[#first_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#func_dbg]], inlinedAt: ![[#]])
// CHECK:       ![[#second_dbg]] = !DILocation(line: [[#]]
// CHECK-SAME:  scope: ![[#func_dbg]], inlinedAt: ![[#]])

#![crate_type = "lib"]

#[no_mangle]
extern "C" fn add_numbers(x: &Option<i32>, y: &Option<i32>) -> i32 {
    let x1 = x.unwrap();
    let y1 = y.unwrap();

    x1 + y1
}
