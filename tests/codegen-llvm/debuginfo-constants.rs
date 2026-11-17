//@ compile-flags: -g -Copt-level=3

// Check that const item with integer types are emitted in the debuginfo.

#![crate_type = "lib"]

pub const A: u8 = 123;
pub const B: i32 = -1;
pub const C: f32 = 1.5;
pub const D: *const () = 123456 as *const ();
pub const UNUSED: u64 = 456;

pub fn test() -> i32 {
    pub const NESTED: f64 = 123.456;
    A as i32 + B + C as i32 + D as i32 + NESTED as i32
}

// CHECK-DAG: ![[EXPR_A:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_A:[0-9]+]], expr: !DIExpression(DW_OP_constu, 123, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_A]] = distinct !DIGlobalVariable(name: "A",{{.*}}line: 7,{{.*}}type: ![[TYPE_A:[0-9]+]]
// CHECK-DAG: ![[TYPE_A]] = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)

// CHECK-DAG: ![[EXPR_B:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_B:[0-9]+]], expr: !DIExpression(DW_OP_constu, 4294967295, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_B]] = distinct !DIGlobalVariable(name: "B",{{.*}}line: 8,{{.*}}type: ![[TYPE_B:[0-9]+]]
// CHECK-DAG: ![[TYPE_B]] = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)

// CHECK-DAG: ![[EXPR_C:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_C:[0-9]+]], expr: !DIExpression(DW_OP_constu, 1069547520, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_C]] = distinct !DIGlobalVariable(name: "C",{{.*}}line: 9,{{.*}}type: ![[TYPE_C:[0-9]+]]
// CHECK-DAG: ![[TYPE_C]] = !DIBasicType(name: "f32", size: 32, encoding: DW_ATE_float)

// CHECK-DAG: ![[EXPR_D:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_D:[0-9]+]], expr: !DIExpression(DW_OP_constu, 123456, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_D]] = distinct !DIGlobalVariable(name: "D",{{.*}}line: 10,{{.*}}type: ![[TYPE_D:[0-9]+]]
// CHECK-DAG: ![[TYPE_D]] = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()"

// CHECK-DAG: ![[EXPR_UNUSED:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_UNUSED:[0-9]+]], expr: !DIExpression(DW_OP_constu, 456, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_UNUSED]] = distinct !DIGlobalVariable(name: "UNUSED",{{.*}}line: 11,{{.*}}type: ![[TYPE_UNUSED:[0-9]+]]
// CHECK-DAG: ![[TYPE_UNUSED]] = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)

// CHECK-DAG: ![[EXPR_NESTED:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VAR_NESTED:[0-9]+]], expr: !DIExpression(DW_OP_constu, 4638387860618067575, DW_OP_stack_value))
// CHECK-DAG: ![[VAR_NESTED]] = distinct !DIGlobalVariable(name: "NESTED",{{.*}}line: 14,{{.*}}type: ![[TYPE_NESTED:[0-9]+]]
// CHECK-DAG: ![[TYPE_NESTED]] = !DIBasicType(name: "f64", size: 64, encoding: DW_ATE_float)

// CHECK-DAG: !DICompileUnit({{.*}}globals: ![[GLOBALS:[0-9]+]]
// CHECK-DAG: ![[GLOBALS]] = !{![[EXPR_A]], ![[EXPR_B]], ![[EXPR_C]], ![[EXPR_D]], ![[EXPR_UNUSED]], ![[EXPR_NESTED]]}
