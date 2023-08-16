//@compile-flags: --error-format=human --cfg a{b}
//@error-in-other-file: invalid `--cfg` argument: `a{b}` (expected `key` or `key="value"`)
fn main() {}
