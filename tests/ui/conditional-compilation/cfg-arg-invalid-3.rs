//@ compile-flags: --cfg a::b

fn main() {}

//~? ERROR invalid `--cfg` argument: `a::b` (argument key must be an identifier)
