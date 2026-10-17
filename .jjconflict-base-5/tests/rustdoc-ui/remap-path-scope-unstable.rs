// Regression test to make sure `--remap-path-scope` is unstable in rustdoc

//@ compile-flags:--remap-path-scope macro

//~? RAW the `-Z unstable-options` flag must also be passed to enable the flag `remap-path-scope`

fn main() {}
