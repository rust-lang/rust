// Test that the fake borrows for matches are removed after borrow checking.

// ignore-wasm32-bare compiled with panic=abort by default

// EMIT_MIR remove_fake_borrows.match_guard.CleanupNonCodegenStatements.diff
fn match_guard(x: Option<&&i32>, c: bool) -> i32 {
    match x {
        Some(0) if c => 0,
        _ => 1,
    }
}

fn main() {
    match_guard(None, true);
}
