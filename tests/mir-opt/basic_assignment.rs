// needs-unwind
// this tests move up progration, which is not yet implemented

// EMIT_MIR basic_assignment.main.ElaborateDrops.diff
// EMIT_MIR basic_assignment.main.SimplifyCfg-initial.after.mir

// Check codegen for assignments (`a = b`) where the left-hand-side is
// not yet initialized. Assignments tend to be absent in simple code,
// so subtle breakage in them can leave a quite hard-to-find trail of
// destruction.

fn main() {
    let nodrop_x = false;
    let nodrop_y;

    // Since boolean does not require drop, this can be a simple
    // assignment:
    nodrop_y = nodrop_x;

    let drop_x: Option<Box<u32>> = None;
    let drop_y;

    // Since the type of `drop_y` has drop, we generate a `replace`
    // terminator:
    drop_y = drop_x;
}
