// check that we don't StorageDead booleans before they are used

// EMIT_MIR issue_38669.main.SimplifyCfg-initial.after.mir
fn main() {
    // CHECK-LABEL: fn main(
    // CHECK: debug should_break => [[should_break:_.*]];
    // CHECK: [[should_break]] = const false;
    // CHECK-NOT: StorageDead([[should_break]])
    // CHECK: switchInt(
    // CHECK-NOT: StorageDead([[should_break]])
    // CHECK: [[should_break]] = const true;
    let mut should_break = false;
    loop {
        if should_break {
            break;
        }
        should_break = true;
    }
}
