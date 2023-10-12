// Verify that `i32::cmp` FnDef type is declared with size 0 and align 1 in LLVM debuginfo.
// compile-flags: -O -g -Cno-prepopulate-passes
// ignore-msvc the types are mangled differently

use std::cmp::Ordering;

fn foo<F: FnOnce(&i32, &i32) -> Ordering>(v1: i32, v2: i32, compare: F) -> Ordering {
    compare(&v1, &v2)
}

pub fn main() {
    foo(0, 1, i32::cmp);
}

// CHECK: %compare.dbg.spill = alloca {}, align 1
// CHECK: call void @llvm.dbg.declare(metadata ptr %compare.dbg.spill, metadata ![[VAR:.*]], metadata !DIExpression()), !dbg !{{.*}}
// CHECK: ![[TYPE:.*]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn(&i32, &i32) -> core::cmp::Ordering", baseType: !{{.*}}, align: 1, dwarfAddressSpace: {{.*}})
// CHECK: ![[VAR]] = !DILocalVariable(name: "compare", scope: !{{.*}}, file: !{{.*}}, line: {{.*}}, type: ![[TYPE]], align: 1)
