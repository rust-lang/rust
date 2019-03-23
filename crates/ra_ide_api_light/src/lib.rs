//! This crate provides those IDE features which use only a single file.
//!
//! This usually means functions which take syntax tree as an input and produce
//! an edit or some auxiliary info.

mod structure;

use ra_syntax::TextRange;

pub use crate::{
    structure::{file_structure, StructureNode},
};
