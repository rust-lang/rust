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

//@ revisions: without_flag with_flag

//@ check-pass
//@ compile-flags: -Zunleash-the-miri-inside-of-you
//@[with_flag] compile-flags: -Awarnings

fn non_constant() {}
const fn constant() { non_constant() }

fn main() {}

//[without_flag]~? WARN skipping const checks
