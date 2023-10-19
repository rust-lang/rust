// skip-filecheck
// Test spanview output (the default value for `-Z dump-mir-spanview` is "statement")
// compile-flags: -Z dump-mir-spanview

// EMIT_MIR spanview_statement.main.built.after.html
fn main() {}
