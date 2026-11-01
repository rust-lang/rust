// ignore-tidy-linelength
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition:future
//@ compile-flags: -Z unstable-options -Z deny-partial-mitigations=garbage

// test that the list of mitigations in the error message is generated correctly

//~? ERROR incorrect value `garbage` for unstable option `deny-partial-mitigations` - comma-separated list of mitigation kinds (available:

fn main() {}
