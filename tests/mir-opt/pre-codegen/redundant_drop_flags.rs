//@ skip-filecheck
//@ compile-flags: -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

enum Container {
    Empty1,
    Empty2,
    Full1(String),
    Full2(String),
}

// From issue #142705
fn redundant_drop_flags(container: Container) -> Option<String> {
    match container {
        Container::Full1(s) | Container::Full2(s) => Some(s),
        Container::Empty1 | Container::Empty2 => None,
        _ => None,
    }
}

fn main() {
    redundant_drop_flags(Container::Empty1);
}

// EMIT_MIR redundant_drop_flags.redundant_drop_flags.GVN.diff
// EMIT_MIR redundant_drop_flags.redundant_drop_flags.JumpThreading.diff
// EMIT_MIR redundant_drop_flags.redundant_drop_flags.runtime-optimized.after.mir
