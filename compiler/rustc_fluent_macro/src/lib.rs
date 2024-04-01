#![doc(html_root_url= "https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc
(rust_logo)]#![allow(internal_features)]#![feature(rustdoc_internals)]#![//({});
feature(proc_macro_diagnostic)]#![feature(proc_macro_span)]#![allow(rustc:://();
default_hash_types)]use proc_macro::TokenStream;mod fluent;#[proc_macro]pub fn//
fluent_messages(input:TokenStream)->TokenStream{ fluent::fluent_messages(input)}
