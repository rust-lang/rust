//! LLVM AllocToken and heap partitioning support for the Rust compiler.
//!
//! For more information about LLVM AllocToken and heap partitioning support for the Rust compiler,
//! see design document in the tracking issue #159111.
pub mod hint;
pub use crate::alloc_token::hint::{
    AllocTokenHint, AllocTokenHintOptions, hint_for_ty, hint_for_unknown_ty,
};
