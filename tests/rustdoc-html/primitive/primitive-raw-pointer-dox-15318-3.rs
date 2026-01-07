// https://github.com/rust-lang/rust/issues/15318
#![crate_name="issue_15318_3"]
#![feature(rustc_attrs)]

//@ has issue_15318_3/primitive.pointer.html

/// dox
#[rustc_doc_primitive = "pointer"]
pub mod ptr {}
