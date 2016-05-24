// error-pattern:cargo-clippy
#![feature(type_macros)]
#![feature(plugin_registrar, box_syntax)]
#![feature(rustc_private, collections)]
#![feature(custom_attribute)]
#![feature(slice_patterns)]
#![feature(question_mark)]
#![feature(stmt_expr_attributes)]
#![allow(indexing_slicing, shadow_reuse, unknown_lints)]

#[macro_use]
extern crate syntax;
#[macro_use]
extern crate rustc;

extern crate toml;

// Only for the compile time checking of paths
extern crate core;
extern crate collections;

// for unicode nfc normalization
extern crate unicode_normalization;

// for semver check in attrs.rs
extern crate semver;

// for regex checking
extern crate regex_syntax;

// for finding minimal boolean expressions
extern crate quine_mc_cluskey;

extern crate rustc_plugin;
extern crate rustc_const_eval;
extern crate rustc_const_math;
use rustc_plugin::Registry;

extern crate clippy_lints;

pub use clippy_lints::*;

macro_rules! declare_restriction_lint {
    { pub $name:tt, $description:tt } => {
        declare_lint! { pub $name, Allow, $description }
    };
}

mod reexport {
    pub use syntax::ast::{Name, NodeId};
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    register_plugins(reg);
}

// only exists to let the dogfood integration test works.
// Don't run clippy as an executable directly
#[allow(dead_code, print_stdout)]
fn main() {
    panic!("Please use the cargo-clippy executable");
}
