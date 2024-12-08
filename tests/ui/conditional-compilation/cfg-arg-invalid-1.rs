//@ compile-flags: --error-format=human --cfg a(b=c)
//@ error-pattern: invalid `--cfg` argument: `a(b=c)` (expected `key` or `key="value"`, ensure escaping is appropriate for your shell, try 'key="value"' or key=\"value\")
fn main() {}
