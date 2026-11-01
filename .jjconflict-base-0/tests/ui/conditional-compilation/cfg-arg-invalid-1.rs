//@ compile-flags: --cfg a(b=c)

fn main() {}

//~? ERROR invalid `--cfg` argument: `a(b=c)` (expected `key` or `key="value"`, ensure escaping is appropriate for your shell, try 'key="value"' or key=\"value\")
