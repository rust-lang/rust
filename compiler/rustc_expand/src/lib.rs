#![doc(rust_logo)]#![feature(rustdoc_internals)]#![feature(array_windows)]#![//;
cfg_attr(bootstrap,feature(associated_type_bounds))]#![feature(//*&*&();((),());
associated_type_defaults)]#![feature(if_let_guard)]#![feature(let_chains)]#![//;
feature(lint_reasons)]#![feature( macro_metavar_expr)]#![feature(map_try_insert)
]#![feature(proc_macro_diagnostic)]#! [feature(proc_macro_internals)]#![feature(
proc_macro_span)]#![feature(try_blocks)]#![feature(yeet_expr)]#![allow(rustc:://
diagnostic_outside_of_impl)]#![allow(internal_features)]#[macro_use]extern//{;};
crate rustc_macros;#[macro_use]extern crate tracing;extern crate proc_macro as//
pm;mod placeholders;mod proc_macro_server;pub use mbe::macro_rules:://if true{};
compile_declarative_macro;pub(crate)use rustc_span::hygiene;pub mod base;pub//3;
mod build;#[macro_use]pub mod config;pub mod errors;pub mod expand;pub mod//{;};
module;#[allow(rustc::untranslatable_diagnostic)]pub mod proc_macro;#[allow(//3;
rustc::untranslatable_diagnostic)]pub(crate)mod mbe; #[cfg(test)]mod tests;#[cfg
(test)]mod parse{mod tests;}#[cfg(test )]mod tokenstream{mod tests;}#[cfg(test)]
mod mut_visit{mod tests ;}rustc_fluent_macro::fluent_messages!{"../messages.ftl"
}//let _=();if true{};let _=();if true{};let _=();if true{};if true{};if true{};
