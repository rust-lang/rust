use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::ty::is_copy;
use itertools::Itertools;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::{Span, Symbol};
use std::fmt;

use super::WRONG_SELF_CONVENTION;
use super::lib::SelfKind;

#[rustfmt::skip]
const CONVENTIONS: [(&[Convention], &[SelfKind]); 9] = [
    (&[Convention::Eq("new")], &[SelfKind::No]),
    (&[Convention::StartsWith("as_")], &[SelfKind::Ref, SelfKind::RefMut]),
    (&[Convention::StartsWith("from_")], &[SelfKind::No]),
    (&[Convention::StartsWith("into_")], &[SelfKind::Value]),
    (&[Convention::StartsWith("is_")], &[SelfKind::RefMut, SelfKind::Ref, SelfKind::No]),
    (&[Convention::Eq("to_mut")], &[SelfKind::RefMut]),
    (&[Convention::StartsWith("to_"), Convention::EndsWith("_mut")], &[SelfKind::RefMut]),

    // Conversion using `to_` can use borrowed (non-Copy types) or owned (Copy types).
    // Source: https://rust-lang.github.io/api-guidelines/naming.html#ad-hoc-conversions-follow-as_-to_-into_-conventions-c-conv
    (&[Convention::StartsWith("to_"), Convention::NotEndsWith("_mut"), Convention::IsSelfTypeCopy(false),
    Convention::IsTraitItem(false), Convention::ImplementsTrait(false)], &[SelfKind::Ref]),
    (&[Convention::StartsWith("to_"), Convention::NotEndsWith("_mut"), Convention::IsSelfTypeCopy(true),
    Convention::IsTraitItem(false), Convention::ImplementsTrait(false)], &[SelfKind::Value]),
];

enum Convention {
    Eq(&'static str),
    StartsWith(&'static str),
    EndsWith(&'static str),
    NotEndsWith(&'static str),
    IsSelfTypeCopy(bool),
    ImplementsTrait(bool),
    IsTraitItem(bool),
}

impl Convention {
    #[must_use]
    fn check<'tcx>(
        &self,
        cx: &LateContext<'tcx>,
        self_ty: Ty<'tcx>,
        other: &str,
        implements_trait: bool,
        is_trait_item: bool,
    ) -> bool {
        match *self {
            Self::Eq(this) => this == other,
            Self::StartsWith(this) => other.starts_with(this) && this != other,
            Self::EndsWith(this) => other.ends_with(this) && this != other,
            Self::NotEndsWith(this) => !Self::EndsWith(this).check(cx, self_ty, other, implements_trait, is_trait_item),
            Self::IsSelfTypeCopy(is_true) => is_true == is_copy(cx, self_ty),
            Self::ImplementsTrait(is_true) => is_true == implements_trait,
            Self::IsTraitItem(is_true) => is_true == is_trait_item,
        }
    }
}

impl fmt::Display for Convention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Self::Eq(this) => write!(f, "`{this}`"),
            Self::StartsWith(this) => write!(f, "`{this}*`"),
            Self::EndsWith(this) => write!(f, "`*{this}`"),
            Self::NotEndsWith(this) => write!(f, "`~{this}`"),
            Self::IsSelfTypeCopy(is_true) => {
                write!(f, "`self` type is{} `Copy`", if is_true { "" } else { " not" })
            },
            Self::ImplementsTrait(is_true) => {
                let (negation, s_suffix) = if is_true { ("", "s") } else { (" does not", "") };
                write!(f, "method{negation} implement{s_suffix} a trait")
            },
            Self::IsTraitItem(is_true) => {
                let suffix = if is_true { " is" } else { " is not" };
                write!(f, "method{suffix} a trait item")
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    item_name: Symbol,
    self_ty: Ty<'tcx>,
    first_arg_ty: Ty<'tcx>,
    first_arg_span: Span,
    implements_trait: bool,
    is_trait_item: bool,
) {
    let item_name_str = item_name.as_str();
    if let Some((conventions, self_kinds)) = &CONVENTIONS.iter().find(|(convs, _)| {
        convs
            .iter()
            .all(|conv| conv.check(cx, self_ty, item_name_str, implements_trait, is_trait_item))
    }) {
        // don't lint if it implements a trait but not willing to check `Copy` types conventions (see #7032)
        if implements_trait
            && !conventions
                .iter()
                .any(|conv| matches!(conv, Convention::IsSelfTypeCopy(_)))
        {
            return;
        }
        if !self_kinds.iter().any(|k| k.matches(cx, self_ty, first_arg_ty)) {
            let suggestion = {
                if conventions.len() > 1 {
                    // Don't mention `NotEndsWith` when there is also `StartsWith` convention present
                    let cut_ends_with_conv = conventions.iter().any(|conv| matches!(conv, Convention::StartsWith(_)))
                        && conventions
                            .iter()
                            .any(|conv| matches!(conv, Convention::NotEndsWith(_)));

                    let s = conventions
                        .iter()
                        .filter(|conv| !(cut_ends_with_conv && matches!(conv, Convention::NotEndsWith(_))))
                        .filter(|conv| !matches!(conv, Convention::ImplementsTrait(_) | Convention::IsTraitItem(_)))
                        .format(" and ");

                    format!("methods with the following characteristics: ({s})")
                } else {
                    format!("methods called {}", &conventions[0])
                }
            };

            span_lint_and_help(
                cx,
                WRONG_SELF_CONVENTION,
                first_arg_span,
                format!(
                    "{suggestion} usually take {}",
                    self_kinds.iter().map(|k| k.description()).format(" or ")
                ),
                None,
                "consider choosing a less ambiguous name",
            );
        }
    }
}
