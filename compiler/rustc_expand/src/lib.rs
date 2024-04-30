// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![doc(rust_logo)]
#![feature(array_windows)]
#![feature(associated_type_defaults)]
#![feature(if_let_guard)]
#![feature(let_chains)]
#![feature(macro_metavar_expr)]
#![feature(map_try_insert)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_internals)]
#![feature(rustdoc_internals)]
#![feature(try_blocks)]
#![feature(yeet_expr)]
// tidy-alphabetical-end

extern crate proc_macro as pm;

mod build;
mod errors;
// FIXME(Nilstrieb) Translate macro_rules diagnostics
#[allow(rustc::untranslatable_diagnostic)]
mod mbe;
mod placeholders;
mod proc_macro_server;

pub use mbe::macro_rules::compile_declarative_macro;
pub mod base;
pub mod config;
pub mod expand;
pub mod module;
// FIXME(Nilstrieb) Translate proc_macro diagnostics
#[allow(rustc::untranslatable_diagnostic)]
pub mod proc_macro;

// HACK(Centril, #64197): These shouldn't really be here.
// Rather, they should be with their respective modules which are defined in other crates.
// However, since for now constructing a `ParseSess` sorta requires `config` from this crate,
// these tests will need to live here in the interim.

#[cfg(test)]
mod tests;
#[cfg(test)]
mod parse {
    mod tests;
}
#[cfg(test)]
mod tokenstream {
    mod tests;
}
#[cfg(test)]
mod mut_visit {
    mod tests;
}

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }
