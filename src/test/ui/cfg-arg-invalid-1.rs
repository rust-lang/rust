// compile-flags: --cfg a(b=c)
// error-pattern: invalid `--cfg` argument: `a(b=c)` (expected `key` or `key="value"`)
fn main() {}
