#![warn(clippy::mod_module_files)]

#[path = "bar/mod.rs"]
pub mod foo;
