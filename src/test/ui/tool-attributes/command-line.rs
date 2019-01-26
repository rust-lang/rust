// compile-pass
// compile-flags:-Z attr-tool=tool1 -Z attr-tool=tool2

#![deny(unused_attributes)]

#[tool1::foo]
fn check() {}

#[tool2::bar]
fn main() {}
