use clippy_utils::last_path_segment;
use if_chain::if_chain;
use rustc_hir::{GenericArg, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_span::source_map::Span;

pub(super) fn match_borrows_parameter(_cx: &LateContext<'_>, qpath: &QPath<'_>) -> Option<Span> {
    let last = last_path_segment(qpath);
    if_chain! {
        if let Some(params) = last.args;
        if !params.parenthesized;
        if let Some(ty) = params.args.iter().find_map(|arg| match arg {
            GenericArg::Type(ty) => Some(ty),
            _ => None,
        });
        if let TyKind::Ref(..) = ty.kind;
        then {
            return Some(ty.span);
        }
    }
    None
}
