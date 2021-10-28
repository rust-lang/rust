#![feature(crate_visibility_modifier)]
#![feature(decl_macro)]
#![feature(destructuring_assignment)]
#![feature(format_args_capture)]
#![feature(if_let_guard)]
#![feature(iter_zip)]
#![feature(let_else)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]
#![feature(try_blocks)]
#![recursion_limit = "256"]

#[macro_use]
extern crate rustc_macros;

extern crate proc_macro as pm;

mod placeholders;
mod proc_macro_server;

pub use mbe::macro_rules::compile_declarative_macro;
crate use rustc_span::hygiene;
pub mod base;
pub mod build;
#[macro_use]
pub mod config;
pub mod expand;
pub mod module;
pub mod proc_macro;

crate mod mbe;

// HACK(Centril, #64197): These shouldn't really be here.
// Rather, they should be with their respective modules which are defined in other crates.
// However, since for now constructing a `ParseSess` sorta requires `config` from this crate,
// these tests will need to live here in the iterim.

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
