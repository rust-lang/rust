//! In this case, `#[rustfmt::skip]` could be interpreted as either:
//! - a module or crate named `rustfmt` that contains an attribute macro
//!  named `skip`
//! - the `skip` attribute of the `rustfmt` tool.
//!
//! In this case there's a module named `rustfmt`, but it doesn't have
//! `skip` in it. Rather than continue iterating through scopes, which
//! could resolve it as the tool attribute, we emit an error.
//!
//! The same applies for the tool explicitly registered with the `register_tool`
//! attribute.
//!
//! See https://github.com/rust-lang/rust/pull/162597#issuecomment-6172442348
//! for the lang decision on this matter.

#![feature(register_tool)]
#![register_tool(custom_tool)]
#![crate_type = "lib"]

mod rustfmt {}

#[rustfmt::skip] //~ ERROR cannot find `skip` in `rustfmt`
struct X;

mod custom_tool {}

#[custom_tool::custom_attr] //~ ERROR cannot find `custom_attr` in `custom_tool`
struct Y;
