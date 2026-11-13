//@ compile-flags: -C panic=abort
//@ test-mir-pass: ElaborateDrops

// Ensures there are no drops for the wildcard match arm.

// EMIT_MIR otherwise_drops.result_ok.ElaborateDrops.diff
fn result_ok(result: Result<String, ()>) -> Option<String> {
    // CHECK-NOT: drop
    match result {
        Ok(s) => Some(s),
        _ => None,
    }
}
