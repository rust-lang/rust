// Test for missing quotes around value, issue #66450.
// compile-flags: --error-format=human --cfg key=value
// error-pattern: invalid `--cfg` argument: `key=value` (expected `key` or `key="value"`, ensure escaping is appropriate for your shell, try 'key="value"' or key=\"value\")
fn main() {}
