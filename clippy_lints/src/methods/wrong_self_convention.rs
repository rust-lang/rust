use crate::methods::SelfKind;
use crate::utils::span_lint;
use rustc_lint::LateContext;
use rustc_middle::ty::TyS;
use rustc_span::source_map::Span;
use std::fmt;

use super::WRONG_PUB_SELF_CONVENTION;
use super::WRONG_SELF_CONVENTION;

#[rustfmt::skip]
const CONVENTIONS: [(Convention, &[SelfKind]); 7] = [
    (Convention::Eq("new"), &[SelfKind::No]),
    (Convention::StartsWith("as_"), &[SelfKind::Ref, SelfKind::RefMut]),
    (Convention::StartsWith("from_"), &[SelfKind::No]),
    (Convention::StartsWith("into_"), &[SelfKind::Value]),
    (Convention::StartsWith("is_"), &[SelfKind::Ref, SelfKind::No]),
    (Convention::Eq("to_mut"), &[SelfKind::RefMut]),
    (Convention::StartsWith("to_"), &[SelfKind::Ref]),
];
enum Convention {
    Eq(&'static str),
    StartsWith(&'static str),
}

impl Convention {
    #[must_use]
    fn check(&self, other: &str) -> bool {
        match *self {
            Self::Eq(this) => this == other,
            Self::StartsWith(this) => other.starts_with(this) && this != other,
        }
    }
}

impl fmt::Display for Convention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        match *self {
            Self::Eq(this) => this.fmt(f),
            Self::StartsWith(this) => this.fmt(f).and_then(|_| '*'.fmt(f)),
        }
    }
}

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    item_name: &str,
    is_pub: bool,
    self_ty: &'tcx TyS<'tcx>,
    first_arg_ty: &'tcx TyS<'tcx>,
    first_arg_span: Span,
) {
    let lint = if is_pub {
        WRONG_PUB_SELF_CONVENTION
    } else {
        WRONG_SELF_CONVENTION
    };
    if let Some((ref conv, self_kinds)) = &CONVENTIONS.iter().find(|(ref conv, _)| conv.check(item_name)) {
        if !self_kinds.iter().any(|k| k.matches(cx, self_ty, first_arg_ty)) {
            span_lint(
                cx,
                lint,
                first_arg_span,
                &format!(
                    "methods called `{}` usually take {}; consider choosing a less ambiguous name",
                    conv,
                    &self_kinds
                        .iter()
                        .map(|k| k.description())
                        .collect::<Vec<_>>()
                        .join(" or ")
                ),
            );
        }
    }
}
