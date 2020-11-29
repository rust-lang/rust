// aux-build:proc-macro-macro.rs
// build-aux-docs
#![deny(broken_intra_doc_links)]

extern crate proc_macro_macro;


pub use proc_macro_macro::{DeriveA, attr_a};
use proc_macro_macro::{DeriveB, attr_b};

// @has proc_macro/struct.Foo.html
// @has - '//a/@href' '../proc_macro/derive.DeriveA.html'
// @has - '//a/@href' '../proc_macro/attr.attr_a.html'
// @has - '//a/@href' '../proc_macro/trait.DeriveTrait.html'
// @has - '//a/@href' '../proc_macro_macro/derive.DeriveB.html'
// @has - '//a/@href' '../proc_macro_macro/attr.attr_b.html'
/// Link to [DeriveA], [attr_a], [DeriveB], [attr_b], [DeriveTrait]
pub struct Foo;

// @has proc_macro/struct.Bar.html
// @has - '//a/@href' '../proc_macro/derive.DeriveA.html'
// @has - '//a/@href' '../proc_macro/attr.attr_a.html'
/// Link to [deriveA](derive@DeriveA) [attr](macro@attr_a)
pub struct Bar;

// this should not cause ambiguity errors
pub trait DeriveTrait {}
