// Issue #825: Should recheck the loop contition after continuing
fn main() { let i = 1; while i > 0 { assert (i > 0); log i; i -= 1; cont; } }
