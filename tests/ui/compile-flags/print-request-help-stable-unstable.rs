//! Check that unstable print requests are omitted from help if compiler is in stable channel.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/138698>

//@ ignore-backends: gcc
//@ compile-flags: --print xxx
//
//@ revisions: stable nightly
//@[stable] act-as-stable
//@[nightly] only-nightly

//~? ERROR unknown print request: `xxx`
