//@ compile-flags: --cfg a=10

fn main() {}

//~? ERROR invalid `--cfg` argument: `a=10` (argument value must be a string)
