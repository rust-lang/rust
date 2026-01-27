//@ compile-flags: --check-cfg 'foo=1x'

fn main() {}

//~? ERROR invalid `--check-cfg` argument
