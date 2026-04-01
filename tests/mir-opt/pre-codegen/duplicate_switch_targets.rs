// skip-filecheck
//@ compile-flags: -O -Zmir-opt-level=2 -Cdebuginfo=0

#![crate_type = "lib"]

pub enum Thing {
    A,
    B,
}

// EMIT_MIR duplicate_switch_targets.ub_if_b.PreCodegen.after.mir
pub unsafe fn ub_if_b(t: Thing) -> Thing {
    match t {
        Thing::A => t,
        Thing::B => std::hint::unreachable_unchecked(),
    }
}
