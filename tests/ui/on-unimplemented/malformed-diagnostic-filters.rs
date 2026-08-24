// Check the lint levels for malformed `rustc_on_unimplemented` filters.

//@ revisions: lint_allow lint_warn lint_deny
//@[lint_allow] check-pass
//@[lint_warn] check-pass

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![cfg_attr(lint_allow, allow(malformed_diagnostic_filters))]
#![cfg_attr(lint_warn, warn(malformed_diagnostic_filters))]
#![cfg_attr(lint_deny, deny(malformed_diagnostic_filters))]

#[rustc_on_unimplemented(on(invalid, message = "unused"), message = "fallback")]
//[lint_warn]~^ WARN invalid flag in `on`-clause
//[lint_deny]~^^ ERROR invalid flag in `on`-clause
trait Trait {}

fn main() {}
