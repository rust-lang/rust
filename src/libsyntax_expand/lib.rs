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
    ($e:expr) => ({
        use std::result::Result::{Ok, Err};
        use errors::FatalError;
        match $e {
            Ok(e) => e,
            Err(mut e) => {
                e.emit();
                FatalError.raise()
            }
        }
    })
}

mod placeholders;
mod proc_macro_server;

crate use syntax_pos::hygiene;
pub use mbe::macro_rules::compile_declarative_macro;
pub mod base;
pub mod build;
pub mod expand;
pub mod proc_macro;

crate mod mbe;
