//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

fn opaque<T>(x: T) -> T {
    x
}

// EMIT_MIR storage.shorten_non_borrowed.MoveElimination.diff
pub fn shorten_non_borrowed(x: u32) -> u32 {
    // This checks that reconstruction can shorten a non-borrowed local's
    // storage to its last use instead of keeping lexical storage markers.
    // CHECK-LABEL: fn shorten_non_borrowed(
    // CHECK: debug a => [[short:_.*]];
    // CHECK: debug b => [[short]];
    // CHECK: StorageLive([[short]]);
    // CHECK: [[short]] = opaque::<u32>(move _1)
    // CHECK: opaque::<u32>(move [[short]]) -> [return: [[short_ret:bb.*]],
    // CHECK: [[short_ret]]: {
    // CHECK-NEXT: StorageDead([[short]]);
    let a = opaque(x);
    let b = a;
    opaque(b)
}

// EMIT_MIR storage.borrowed_not_shortened_to_last_direct_use.MoveElimination.diff
pub fn borrowed_not_shortened_to_last_direct_use(x: u32) -> u32 {
    // This checks that a borrowed local is not shortened merely to its last
    // direct use; the borrow keeps its storage live while the reference exists.
    // CHECK-LABEL: fn borrowed_not_shortened_to_last_direct_use(
    // CHECK: debug a => _1;
    // CHECK: debug r => [[borrow_ref:_.*]];
    // CHECK: debug out => [[borrow_out:_.*]];
    // CHECK: StorageLive([[borrow_ref]]);
    // CHECK-NEXT: [[borrow_ref]] = &_1;
    // CHECK: StorageLive([[borrow_out]]);
    // CHECK-NEXT: [[borrow_out]] = copy _1;
    // CHECK: copy (*[[borrow_ref]]);
    // CHECK-NEXT: StorageDead([[borrow_ref]]);
    let a = x;
    let r = &a;
    let out = a;
    opaque(*r + out)
}

// EMIT_MIR storage.storage_live_moved_to_branch.MoveElimination.diff
pub fn storage_live_moved_to_branch(flag: bool) {
    // This checks storage reconstruction can shrink a local declared before a
    // branch so its storage is live only on the arm where it is initialized.
    // CHECK-LABEL: fn storage_live_moved_to_branch(
    // CHECK: debug x => [[branch_tmp:_.*]];
    // CHECK: switchInt(move _1) -> [0: bb3, otherwise: bb1];
    // CHECK: bb1: {
    // CHECK: StorageLive([[branch_tmp]]);
    // CHECK: [[branch_tmp]] = const 1_u32;
    // CHECK: opaque::<u32>(move [[branch_tmp]]) -> [return: [[branch_ret:bb.*]],
    // CHECK: [[branch_ret]]: {
    // CHECK: StorageDead([[branch_tmp]]);
    let x: u32;
    if flag {
        x = 1;
        opaque(x);
    }
}

// EMIT_MIR storage.address_observed_storage_dead_at_end.MoveElimination.diff
pub fn address_observed_storage_dead_at_end(flag: bool) {
    // This checks that an address-observed local declared before a branch still
    // has StorageLive moved into the initialized arm, but StorageDead remains at
    // the end of the function instead of being shortened to the last direct use.
    // CHECK-LABEL: fn address_observed_storage_dead_at_end(
    // CHECK: debug x => [[addr_tmp:_.*]];
    // CHECK: switchInt(move _1) -> [0: [[skip:bb.*]], otherwise: [[init:bb.*]]];
    // CHECK: [[init]]: {
    // CHECK: StorageLive([[addr_tmp]]);
    // CHECK: [[addr_tmp]] = const 1_u32;
    // CHECK: &raw const [[addr_tmp]];
    // CHECK: opaque::<*const u32>
    // CHECK-NOT: StorageDead([[addr_tmp]]);
    // CHECK: [[skip]]: {
    // CHECK: StorageLive([[addr_tmp]]);
    // CHECK: {{bb.*}}: {
    // CHECK: StorageDead([[addr_tmp]]);
    let x: u32;
    if flag {
        x = 1;
        opaque(&raw const x);
    }
}

// EMIT_MIR storage.terminator_end_storage_dead_in_successor.MoveElimination.diff
pub fn terminator_end_storage_dead_in_successor(x: u32) -> u32 {
    // This checks storage reconstruction when the last use of a local is as a
    // call argument in a terminator.
    // CHECK-LABEL: fn terminator_end_storage_dead_in_successor(
    // CHECK: debug tmp => [[term_tmp:_.*]];
    // CHECK: StorageLive([[term_tmp]]);
    // CHECK: [[term_tmp]] = opaque::<u32>(move _1)
    // CHECK: opaque::<u32>(move [[term_tmp]]) -> [return: [[term_ret:bb.*]],
    // CHECK: [[term_ret]]: {
    // CHECK-NEXT: StorageDead([[term_tmp]]);
    let tmp = opaque(x);
    let out = opaque(tmp);
    out
}

// EMIT_MIR storage.critical_edge_split_for_storage_live.MoveElimination.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn critical_edge_split_for_storage_live(flag: bool) {
    // This checks storage reconstruction on a custom CFG where the `false`
    // branch goes directly to the final block. Since that block also has an
    // incoming edge from the initialized branch, inserting StorageLive precisely
    // requires splitting the critical edge from the entry switch.
    // CHECK-LABEL: fn critical_edge_split_for_storage_live(
    // CHECK: debug x => [[crit_tmp:_.*]];
    // CHECK: switchInt(copy _1) -> [1: [[init:bb.*]], otherwise: [[split:bb.*]]];
    // CHECK: [[init]]: {
    // CHECK: StorageLive([[crit_tmp]]);
    // CHECK: [[crit_tmp]] = const 1_u32;
    // CHECK: &raw const [[crit_tmp]];
    // CHECK: opaque::<*const u32>{{.*}} -> [return: [[ret:bb.*]],
    // CHECK: [[ret:bb.*]]: {
    // CHECK: StorageDead([[crit_tmp]]);
    // CHECK: [[split]]: {
    // CHECK-NEXT: StorageLive([[crit_tmp]]);
    // CHECK-NEXT: goto -> [[ret]];
    mir! {
        let x: u32;
        let ptr: *const u32;
        debug x => x;

        {
            match flag {
                true => init,
                _ => ret,
            }
        }

        init = {
            StorageLive(x);
            x = 1;
            ptr = &raw const x;
            Call(ptr = opaque::<*const u32>(ptr), ReturnTo(ret), UnwindUnreachable())
        }

        ret = {
            StorageDead(x);
            Return()
        }
    }
}
