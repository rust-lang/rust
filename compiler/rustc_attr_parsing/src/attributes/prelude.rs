// data structures
#[doc(hidden)]
pub(super) use rustc_feature::{AttributeTemplate, template};
#[doc(hidden)]
pub(super) use rustc_hir::attrs::AttributeKind;
#[doc(hidden)]
pub(super) use rustc_hir::lints::AttributeLintKind;
#[doc(hidden)]
pub(super) use rustc_hir::{MethodKind, Target};
#[doc(hidden)]
pub(super) use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
#[doc(hidden)]
pub(super) use thin_vec::ThinVec;

#[doc(hidden)]
pub(super) use crate::attributes::{
    AcceptMapping, AttributeOrder, AttributeParser, CombineAttributeParser, ConvertFn,
    NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
// contexts
#[doc(hidden)]
pub(super) use crate::context::{AcceptContext, FinalizeContext, Stage};
#[doc(hidden)]
pub(super) use crate::parser::*;
// target checking
#[doc(hidden)]
pub(super) use crate::target_checking::Policy::{Allow, Error, Warn};
#[doc(hidden)]
pub(super) use crate::target_checking::{ALL_TARGETS, AllowedTargets};
