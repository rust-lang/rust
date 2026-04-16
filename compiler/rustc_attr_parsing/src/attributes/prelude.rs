// data structures
#[doc(hidden)]
pub(super) use rustc_feature::{AttributeGate, AttributeGate::*, AttributeTemplate, template};
#[doc(hidden)]
pub(super) use rustc_hir::attrs::AttributeKind;
#[doc(hidden)]
pub(super) use rustc_hir::{MethodKind, Target};
#[doc(hidden)]
pub(super) use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
#[doc(hidden)]
pub(super) use thin_vec::ThinVec;

#[doc(hidden)]
pub(super) use crate::attributes::{
    AcceptMapping, AttributeParser, AttributeSafety, CombineAttributeParser, ConvertFn,
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
#[doc(hidden)]
pub(super) use crate::{experimental, gated, gated_rustc_attr};
