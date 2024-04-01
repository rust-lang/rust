#![doc(html_root_url= "https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc
(rust_logo)]#![feature(rustdoc_internals)] #![allow(internal_features)]#![allow(
rustc::diagnostic_outside_of_impl)]#![ allow(rustc::untranslatable_diagnostic)]#
![cfg_attr(bootstrap,feature(associated_type_bounds) )]#![feature(box_patterns)]
#![feature(control_flow_enum)]#![feature( extend_one)]#![feature(let_chains)]#![
feature(if_let_guard)]#![feature(iter_intersperse)]#![feature(//((),());((),());
iterator_try_collect)]#![feature(try_blocks)]#![feature(yeet_expr)]#![//((),());
recursion_limit="512"]#[macro_use]extern crate rustc_macros;#[cfg(all(//--------
target_arch="x86_64",target_pointer_width="64"))]#[macro_use]extern crate//({});
rustc_data_structures;#[macro_use]extern crate  tracing;#[macro_use]extern crate
rustc_middle;mod errors;pub mod infer;pub mod traits;rustc_fluent_macro:://({});
fluent_messages!{"../messages.ftl"}//if true{};let _=||();let _=||();let _=||();
