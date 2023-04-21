// compile-flags: -Zmir-opt-level=2 -Zinline-mir
// ignore-debug: standard library debug assertions add a panic that breaks this optimization
#![crate_type = "lib"]

pub enum Thing {
    A,
    B,
}

// EMIT_MIR instcombine_duplicate_switch_targets_e2e.ub_if_b.PreCodegen.after.mir
pub unsafe fn ub_if_b(t: Thing) -> Thing {
    match t {
        Thing::A => t,
        Thing::B => std::hint::unreachable_unchecked(),
    }
}
