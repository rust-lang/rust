// ignore-tidy-linelength
//@ check-fail
//@ ignore-nvptx64 stack protector is not supported
//@ ignore-wasm32-unknown-unknown stack protector is not supported
//@ edition: 2024
//@ compile-flags: -Z allow-partial-mitigations=!control-flow-guard -C control-flow-guard=on

// check that in edition 2024, it is still possible to explicitly
// disallow partial mitigations (in edition=future, they are
// disallowed by default)

fn main() {}
//~^ ERROR that is not protected by
//~| ERROR that is not protected by
//~| ERROR that is not protected by
//~| ERROR that is not protected by
//~| ERROR that is not protected by
