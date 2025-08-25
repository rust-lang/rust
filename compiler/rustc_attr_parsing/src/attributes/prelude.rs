// parsing
// templates
pub(super) use rustc_feature::{AttributeTemplate, template};
// data structures
pub(super) use rustc_hir::attrs::AttributeKind;
pub(super) use rustc_hir::lints::AttributeLintKind;
pub(super) use rustc_hir::{MethodKind, Target};
pub(super) use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
pub(super) use thin_vec::ThinVec;

pub(super) use crate::attributes::{
    AcceptMapping, AttributeOrder, AttributeParser, CombineAttributeParser, ConvertFn,
    NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
// contexts
pub(super) use crate::context::{AcceptContext, FinalizeContext, Stage};
pub(super) use crate::parser::*;
// target checking
pub(super) use crate::target_checking::Policy::{Allow, Error, Warn};
pub(super) use crate::target_checking::{ALL_TARGETS, AllowedTargets};
