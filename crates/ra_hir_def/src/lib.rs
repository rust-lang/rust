//! `ra_hir_def` contains initial "phases" of the compiler. Roughly, everything
//! before types.
//!
//! Note that we are in the process of moving parts of `ra_hir` into
//! `ra_hir_def`, so this crates doesn't contain a lot at the moment.

pub mod db;

pub mod ast_id_map;

pub mod expand;
