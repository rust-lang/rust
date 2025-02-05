// ignore-tidy-linelength
//! Check that `-A warnings` cli flag applies to non-lint warnings as well.
//!
//! This test tries to exercise that by checking that a non-lint warning (normally
//! warn-by-default) is suppressed if the `-A warnings` cli flag is passed.
//!
//! Take special note that `warnings` is a special pseudo lint group in relationship to non-lint
//! warnings, which is somewhat special. This test does not exercise other `-A <other_lint_group>`
//! that check that they are working in the same way, only `warnings` specifically.
//!
//! # Relevant context
//!
//! - Original impl PR: <https://github.com/rust-lang/rust/pull/21248>.
//! - RFC 507 "Release channels":
//!   <https://github.com/rust-lang/rfcs/blob/c017755b9bfa0421570d92ba38082302e0f3ad4f/text/0507-release-channels.md>.
#![feature(rustc_attrs)]

//@ revisions: without_flag with_flag

//@[with_flag] compile-flags: -Awarnings

//@ check-pass

#[rustc_error(warn)]
fn main() {}
//[without_flag]~^ WARN unexpected annotation used with `#[rustc_error(...)]`!
