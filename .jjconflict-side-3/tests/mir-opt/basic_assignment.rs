//@ test-mir-pass: ElaborateDrops
//@ needs-unwind
// this tests move up progration, which is not yet implemented

// EMIT_MIR basic_assignment.main.ElaborateDrops.diff
// EMIT_MIR basic_assignment.main.SimplifyCfg-initial.after.mir

// Check codegen for assignments (`a = b`) where the left-hand-side is
// not yet initialized. Assignments tend to be absent in simple code,
// so subtle breakage in them can leave a quite hard-to-find trail of
// destruction.

fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug nodrop_x => [[nodrop_x:_.*]];
    // CHECK: debug nodrop_y => [[nodrop_y:_.*]];
    // CHECK: debug drop_x => [[drop_x:_.*]];
    // CHECK: debug drop_y => [[drop_y:_.*]];
    // CHECK-NOT: drop([[nodrop_x]])
    // CHECK-NOT: drop([[nodrop_y]])
    // CHECK-NOT: drop([[drop_x]])
    // CHECK: [[drop_tmp:_.*]] = move [[drop_x]];
    // CHECK-NOT: drop([[drop_x]])
    // CHECK-NOT: drop([[drop_tmp]])
    // CHECK: [[drop_y]] = move [[drop_tmp]];
    // CHECK-NOT: drop([[drop_x]])
    // CHECK-NOT: drop([[drop_tmp]])
    // CHECK: drop([[drop_y]])
    // CHECK-NOT: drop([[drop_x]])
    // CHECK-NOT: drop([[drop_tmp]])
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
