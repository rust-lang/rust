//@ compile-flags: --cfg )

fn main() {}

//~? ERROR invalid `--cfg` argument: `)` (expected `key` or `key="value"`)
