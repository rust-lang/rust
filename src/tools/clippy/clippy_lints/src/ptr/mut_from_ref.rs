use super::MUT_FROM_REF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::visitors::contains_unsafe_block;
use rustc_errors::MultiSpan;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{self as hir, Body, FnRetTy, FnSig, GenericArg, Lifetime, Mutability, TyKind};
use rustc_lint::LateContext;
use rustc_span::Span;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, sig: &FnSig<'_>, body: Option<&Body<'tcx>>) {
    let FnRetTy::Return(ty) = sig.decl.output else { return };
    for (out, mutability, out_span) in get_lifetimes(ty) {
        if mutability != Some(Mutability::Mut) {
            continue;
        }
        let out_region = cx.tcx.named_bound_var(out.hir_id);
        // `None` if one of the types contains `&'a mut T` or `T<'a>`.
        // Else, contains all the locations of `&'a T` types.
        let args_immut_refs: Option<Vec<Span>> = sig
            .decl
            .inputs
            .iter()
            .flat_map(get_lifetimes)
            .filter(|&(lt, _, _)| cx.tcx.named_bound_var(lt.hir_id) == out_region)
            .map(|(_, mutability, span)| (mutability == Some(Mutability::Not)).then_some(span))
            .collect();
        if let Some(args_immut_refs) = args_immut_refs
            && !args_immut_refs.is_empty()
            && body.is_none_or(|body| sig.header.is_unsafe() || contains_unsafe_block(cx, body.value))
        {
            span_lint_and_then(
                cx,
                MUT_FROM_REF,
                out_span,
                "mutable borrow from immutable input(s)",
                |diag| {
                    let ms = MultiSpan::from_spans(args_immut_refs);
                    diag.span_note(ms, "immutable borrow here");
                },
            );
        }
    }
}

struct LifetimeVisitor<'tcx> {
    result: Vec<(&'tcx Lifetime, Option<Mutability>, Span)>,
}

impl<'tcx> Visitor<'tcx> for LifetimeVisitor<'tcx> {
    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) {
        if let TyKind::Ref(lt, ref m) = ty.kind {
            self.result.push((lt, Some(m.mutbl), ty.span));
        }
        hir::intravisit::walk_ty(self, ty);
    }

    fn visit_generic_arg(&mut self, generic_arg: &'tcx GenericArg<'tcx>) {
        if let GenericArg::Lifetime(lt) = generic_arg {
            self.result.push((lt, None, generic_arg.span()));
        }
        hir::intravisit::walk_generic_arg(self, generic_arg);
    }
}

/// Visit `ty` and collect the all the lifetimes appearing in it, implicit or not.
///
/// The second field of the vector's elements indicate if the lifetime is attached to a
/// shared reference, a mutable reference, or neither.
fn get_lifetimes<'tcx>(ty: &'tcx hir::Ty<'tcx>) -> Vec<(&'tcx Lifetime, Option<Mutability>, Span)> {
    use hir::intravisit::VisitorExt as _;

    let mut visitor = LifetimeVisitor { result: Vec::new() };
    visitor.visit_ty_unambig(ty);
    visitor.result
}
