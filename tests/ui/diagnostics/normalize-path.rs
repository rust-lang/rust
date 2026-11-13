// Check that diagnostic file paths are lexically normalized:
// the error below points at `auxiliary/helper.rs`, not `auxiliary/sub/../helper.rs`.
// See #51349.
#[path = "auxiliary/sub/mod.rs"]
mod sub;

fn main() {}

//~? ERROR mismatched types
