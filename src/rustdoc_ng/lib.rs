#[link(name = "rustdoc_ng",
       vers = "0.1.0",
       uuid = "8c6e4598-1596-4aa5-a24c-b811914bbbc6")];
#[desc = "rustdoc, the Rust documentation extractor"];
#[license = "MIT/ASL2"];
#[crate_type = "lib"];

#[deny(warnings)];

extern mod syntax;
extern mod rustc;

extern mod extra;

use extra::serialize::Encodable;

pub mod core;
pub mod doctree;
pub mod clean;
pub mod visit_ast;
pub mod fold;
pub mod plugins;
pub mod passes;

pub static SCHEMA_VERSION: &'static str = "0.8.0";

pub static ctxtkey: std::local_data::Key<@core::DocContext> = &std::local_data::Key;
