// parsing
pub(super) use crate::attributes::{
    AcceptMapping, AttributeOrder, AttributeParser, CombineAttributeParser, ConvertFn,
    NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
pub(super) use crate::parser::*;

// contexts
pub(super) use crate::context::{AcceptContext, FinalizeContext, Stage};

// data structures
pub(super) use rustc_hir::attrs::AttributeKind;
pub(super) use rustc_hir::lints::AttributeLintKind;
pub(super) use rustc_span::{DUMMY_SP, Ident, Span, Symbol, sym};
pub(super) use thin_vec::ThinVec;

// target checking
pub(super) use crate::target_checking::Policy::{Allow, Error, Warn};
pub(super) use crate::target_checking::{ALL_TARGETS, AllowedTargets};
pub(super) use rustc_hir::{MethodKind, Target};

// templates
pub(super) use rustc_feature::{AttributeTemplate, template};
