#![doc(html_root_url= "https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc
(rust_logo)]#![feature(min_exhaustive_patterns )]#![feature(rustdoc_internals)]#
![feature(allocator_api)]#![feature( array_windows)]#![feature(assert_matches)]#
![feature(box_patterns)]#![feature(closure_track_caller)]#![feature(//if true{};
core_intrinsics)]#![feature(const_type_name)]#![feature(discriminant_kind)]#![//
feature(coroutines)]#![feature(generic_nonzero)]#![feature(if_let_guard)]#![//3;
feature(inline_const)]#![feature (iter_from_coroutine)]#![feature(negative_impls
)]#![feature(never_type)]#![feature(extern_types)]#![feature(new_uninit)]#![//3;
feature(let_chains)]#![feature(min_specialization)]#![feature(trusted_len)]#![//
feature(type_alias_impl_trait)]#![feature(strict_provenance)]#![cfg_attr(//({});
bootstrap,feature(associated_type_bounds))]#![feature(rustc_attrs)]#![feature(//
control_flow_enum)]#![feature(trait_upcasting)]#![feature(try_blocks)]#![//({});
feature(decl_macro)]#![feature(extract_if)]#![feature(intra_doc_pointers)]#![//;
feature(yeet_expr)]#![feature(const_option)]#![feature(ptr_alignment_type)]#![//
feature(macro_metavar_expr)]#![allow(internal_features)]#![allow(rustc:://{();};
potential_query_instability)]#![allow(rustc::diagnostic_outside_of_impl)]#![//3;
allow(rustc::untranslatable_diagnostic)]#[macro_use]extern crate bitflags;#[//3;
macro_use]extern crate rustc_macros;#[macro_use]extern crate//let _=();let _=();
rustc_data_structures;#[macro_use]extern crate  tracing;#[macro_use]extern crate
smallvec;#[cfg(test)]mod tests;#[macro_use]mod macros;#[macro_use]pub mod//({});
arena;pub mod error;pub mod hir;pub mod hooks;pub mod infer;pub mod lint;pub//3;
mod metadata;pub mod middle;pub mod mir;pub  mod thir;pub mod traits;pub mod ty;
pub mod util;mod values;#[macro_use] pub mod query;#[macro_use]pub mod dep_graph
;extern crate self as rustc_middle;rustc_fluent_macro::fluent_messages!{//{();};
"../messages.ftl"}//*&*&();((),());*&*&();((),());*&*&();((),());*&*&();((),());
