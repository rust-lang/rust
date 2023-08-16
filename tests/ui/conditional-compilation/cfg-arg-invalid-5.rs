//@compile-flags: --cfg a=10
//@error-in-other-file: invalid `--cfg` argument: `a=10` (argument value must be a string)
fn main() {}
