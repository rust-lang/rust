//! Unfortunately the development of `#[phase]` and `#[no_link]`
//! predates Zulip, and thus has been lost in the sands of time.
//! Understanding the true nature of this test has been left as
//! an exercise for the reader.
//!
//! But we guess from the git history that originally this
//! test was supposed to check that we error if we can't find
//! an extern crate annotated with `#[phase(syntax)]`,
//! see `macro-crate-unknown-crate.rs` in
//! <https://github.com/rust-lang/rust/pull/11151>. Later, we changed
//! `#[phase]` to `#![feature(plugin)]` and added a `#[no_link]`.
//!
//! I suppose that this now tests that we still error if we can't
//! find a `#[no_link]` extern crate?

#[no_link]
extern crate doesnt_exist; //~ ERROR can't find crate

fn main() {}
