#![feature(crate_visibility_modifier)]
#![feature(decl_macro)]
#![feature(proc_macro_diagnostic)]
#![feature(proc_macro_internals)]
#![feature(proc_macro_span)]

extern crate proc_macro as pm;

// A variant of 'try!' that panics on an Err. This is used as a crutch on the
// way towards a non-panic!-prone parser. It should be used for fatal parsing
// errors; eventually we plan to convert all code using panictry to just use
// normal try.
#[macro_export]
macro_rules! panictry {
    ($e:expr) => {{
        use rustc_errors::FatalError;
        use std::result::Result::{Err, Ok};
        match $e {
            Ok(e) => e,
            Err(mut e) => {
                e.emit();
                FatalError.raise()
            }
        }
    }};
}

mod placeholders;
mod proc_macro_server;

pub use mbe::macro_rules::compile_declarative_macro;
crate use rustc_span::hygiene;
pub mod base;
pub mod build;
pub mod expand;
pub use rustc_parse::config;
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
    #[cfg(test)]
    mod tests;
    #[cfg(test)]
    mod lexer {
        #[cfg(test)]
        mod tests;
    }
}
#[cfg(test)]
mod tokenstream {
    #[cfg(test)]
    mod tests;
}
#[cfg(test)]
mod mut_visit {
    #[cfg(test)]
    mod tests;
}
