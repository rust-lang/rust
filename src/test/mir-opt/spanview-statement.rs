// Test spanview output (the default value for `-Z dump-mir-spanview` is "statement")
// compile-flags: -Z dump-mir-spanview

// EMIT_MIR spanview_statement.main.mir_map.0.html
fn main() {}
