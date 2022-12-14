#![feature(lint_reasons)]
#![deny(clippy::allow_attributes_without_reason)]

// These should trigger the lint
#[allow(dead_code)]
#[allow(dead_code, deprecated)]
// These should be fine
#[allow(dead_code, reason = "This should be allowed")]
#[warn(dyn_drop, reason = "Warnings can also have reasons")]
#[warn(deref_nullptr)]
#[deny(deref_nullptr)]
#[forbid(deref_nullptr)]

fn main() {}
