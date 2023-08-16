//@compile-flags: --cfg a::b
//@error-in-other-file: invalid `--cfg` argument: `a::b` (argument key must be an identifier)
fn main() {}
