// compile-flags: --cfg a{b}
// error-pattern: invalid `--cfg` argument: `a{b}` (expected `key` or `key="value"`, ensure escaping is appropriate for your shell, try 'key="value"' or key=\"value\")
fn main() {}
