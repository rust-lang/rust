//! HIR datatypes. See the [rustc dev guide] for more info.
//!
//! [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html

// tidy-alphabetical-start
#![cfg_attr(bootstrap, feature(never_type))]
#![feature(associated_type_defaults)]
#![feature(closure_track_caller)]
#![feature(const_default)]
#![feature(const_trait_impl)]
#![feature(default_field_values)]
#![feature(derive_const)]
#![feature(exhaustive_patterns)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

mod arena;
pub mod def;
mod hir;
pub mod intravisit;
pub mod lints;
pub mod pat_util;
mod stable_hash_impls;
mod target_impls;

#[doc(no_inline)]
pub use hir::*;
pub use rustc_attr_ir::{self as attrs, find_attr};
pub use rustc_hir_id::*;
pub use rustc_span::def_id;
// FIXME: Remove this use tree, replace by `rustc_hir::attrs` or `rustc_attr_ir` imports
#[doc(hidden)]
pub use {
    attrs::target::{self, AssocCtxt, MethodKind, Target},
    attrs::{
        AttrArgs, AttrItem, AttrPath, Attribute, ConstStability, DefaultBodyStability,
        HashIgnoredAttrId, PartialConstStability, Stability, StabilityLevel, StableSince,
        UnstableReason, VERSION_PLACEHOLDER,
    },
};

pub use crate::arena::Arena;
