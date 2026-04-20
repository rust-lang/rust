// Verify that diagnostic file paths are lexically normalized.
// Without the fix for #51349, the error location would show
// `auxiliary/sub/../helper.rs` instead of `auxiliary/helper.rs`.
#[path = "auxiliary/sub/mod.rs"]
mod sub;

fn main() {}

//~? ERROR mismatched types
