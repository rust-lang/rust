// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
// Test that we correctly generate StorageDead statements for while loop
// conditions on all branches
//@ compile-flags: -Zmir-opt-level=0

fn get_bool(c: bool) -> bool {
    c
}

// EMIT_MIR while_storage.while_loop.PreCodegen.after.mir
fn while_loop(c: bool) {
    // CHECK-LABEL: fn while_loop(
    // CHECK: bb0: {
    // CHECK-NEXT:     goto -> bb1;
    // CHECK: bb1: {
    // CHECK-NEXT:     StorageLive(_3);
    // CHECK-NEXT:     StorageLive(_2);
    // CHECK-NEXT:     _2 = copy _1;
    // CHECK-NEXT:     _3 = get_bool(move _2) -> [return: bb2, unwind
    // CHECK: bb2: {
    // CHECK-NEXT:     switchInt(move _3) -> [0: bb3, otherwise: bb4];
    // CHECK: bb3: {
    // CHECK-NEXT:     StorageDead(_2);
    // CHECK-NEXT:     StorageLive(_9);
    // CHECK-NEXT:     _0 = const ();
    // CHECK-NEXT:     StorageDead(_9);
    // CHECK-NEXT:     goto -> bb8;
    // CHECK: bb4: {
    // CHECK-NEXT:     StorageDead(_2);
    // CHECK-NEXT:     StorageLive(_5);
    // CHECK-NEXT:     StorageLive(_4);
    // CHECK-NEXT:     _4 = copy _1;
    // CHECK-NEXT:     _5 = get_bool(move _4) -> [return: bb5, unwind
    // CHECK: bb5: {
    // CHECK-NEXT:     switchInt(move _5) -> [0: bb6, otherwise: bb7];
    // CHECK: bb6: {
    // CHECK-NEXT:     StorageDead(_4);
    // CHECK-NEXT:     _6 = const ();
    // CHECK-NEXT:     StorageDead(_5);
    // CHECK-NEXT:     StorageDead(_3);
    // CHECK-NEXT:     goto -> bb1;
    // CHECK: bb7: {
    // CHECK-NEXT:     StorageDead(_4);
    // CHECK-NEXT:     _0 = const ();
    // CHECK-NEXT:     StorageDead(_5);
    // CHECK-NEXT:     goto -> bb8;
    // CHECK: bb8: {
    // CHECK-NEXT:     StorageDead(_3);
    // CHECK-NEXT:     return;

    while get_bool(c) {
        if get_bool(c) {
            break;
        }
    }
}

fn main() {
    while_loop(false);
}
