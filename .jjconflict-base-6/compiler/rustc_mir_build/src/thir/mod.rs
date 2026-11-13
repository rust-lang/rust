//! The MIR is built from some typed high-level IR
//! (THIR). This section defines the THIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

pub(crate) mod constant;
pub(crate) mod cx;
pub(crate) mod pattern;
pub mod print;
mod util;
