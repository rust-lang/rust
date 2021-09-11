//! The MIR is built from some typed high-level IR
//! (THIR). This section defines the THIR along with a trait for
//! accessing it. The intention is to allow MIR construction to be
//! unit-tested and separated from the Rust source and compiler data
//! structures.

crate mod constant;

crate mod cx;

crate mod pattern;

mod util;
