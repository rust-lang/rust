// Verify that `i32::cmp` FnDef type is declared with a size of 0 and an
// alignment of 8 bits (1 byte) in LLVM debuginfo.

//@ compile-flags: -Copt-level=3 -g -Cno-prepopulate-passes
//@ ignore-msvc the types are mangled differently

use std::cmp::Ordering;

fn foo<F: FnOnce(&i32, &i32) -> Ordering>(v1: i32, v2: i32, compare: F) -> Ordering {
    compare(&v1, &v2)
}

pub fn main() {
    foo(0, 1, i32::cmp);
}

// CHECK: %compare.dbg.spill = alloca [0 x i8], align 1
// CHECK: dbg{{.}}declare({{(metadata )?}}ptr %compare.dbg.spill, {{(metadata )?}}![[VAR:.*]], {{(metadata )?}}!DIExpression()
// CHECK-DAG: ![[TYPE:.*]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "fn(&i32, &i32) -> core::cmp::Ordering", baseType: !{{.*}}, align: 8, dwarfAddressSpace: {{.*}})
// CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "compare", scope: !{{.*}}, file: !{{.*}}, line: {{.*}}, type: ![[TYPE]], align: 8)
