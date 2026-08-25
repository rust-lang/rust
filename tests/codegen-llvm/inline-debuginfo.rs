#![crate_type = "rlib"]
//@ compile-flags: -Copt-level=3 -g
//

#[no_mangle]
#[inline(always)]
pub extern "C" fn callee(x: u32) -> u32 {
    x + 4
}

// CHECK-LABEL: caller
// CHECK: dbg{{.}}value({{(metadata )?}}i32 %y, {{(metadata )?}}!{{.*}}, {{(metadata )?}}!DIExpression(DW_OP_constu, 3, DW_OP_minus, DW_OP_stack_value){{.*}} [[A:![0-9]+]]
// CHECK: [[A]] = !DILocation(line: {{.*}}, scope: {{.*}}, inlinedAt: {{.*}})
#[no_mangle]
pub extern "C" fn caller(y: u32) -> u32 {
    callee(y - 3)
}
