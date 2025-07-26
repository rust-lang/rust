//@ compile-flags: -Copt-level=0 -Zmir-opt-level=1 -Cdebuginfo=limited
//@ edition: 2024
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]
#![feature(try_blocks)]

// EMIT_MIR option_bubble_debug.option_direct.PreCodegen.after.mir
pub fn option_direct(x: Option<u32>) -> Option<u32> {
    // CHECK-LABEL: fn option_direct(_1: Option<u32>) -> Option<u32>
    // CHECK: = discriminant(_1);
    // CHECK: [[TEMP:_.+]] = Not({{.+}});
    // CHECK: _0 = Option::<u32>::Some(move [[TEMP]]);

    match x {
        Some(x) => Some(!x),
        None => None,
    }
}

// EMIT_MIR option_bubble_debug.option_traits.PreCodegen.after.mir
pub fn option_traits(x: Option<u32>) -> Option<u32> {
    // CHECK-LABEL: fn option_traits(_1: Option<u32>) -> Option<u32>
    // CHECK: = <Option<u32> as Try>::branch(copy _1)
    // CHECK: [[TEMP:_.+]] = Not({{.+}});
    // CHECK: _0 = <Option<u32> as Try>::from_output(move [[TEMP]])

    try { !(x?) }
}
