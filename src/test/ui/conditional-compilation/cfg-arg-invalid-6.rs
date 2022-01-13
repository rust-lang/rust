// compile-flags: --cfg a{
// error-pattern: invalid `--cfg` argument: `a{` (expected `key` or `key="value"`, ensure escaping is appropriate for your shell, try 'key="value"' or key=\"value\")
fn main() {}
