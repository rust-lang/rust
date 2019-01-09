// compile-flags: --cfg a{b}
// error-pattern: invalid `--cfg` argument: `a{b}` (expected `key` or `key="value"`)
fn main() {}
