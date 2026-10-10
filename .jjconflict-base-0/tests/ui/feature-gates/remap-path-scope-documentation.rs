// Test that documentation scope is unstable

//@ compile-flags: --remap-path-scope=documentation

//~? ERROR remapping `documentation` path scope requested but `-Zunstable-options` not specified

fn main() {}
