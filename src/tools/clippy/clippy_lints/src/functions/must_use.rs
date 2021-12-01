use rustc_ast::ast::Attribute;
use rustc_errors::Applicability;
use rustc_hir::def_id::{DefIdSet, LocalDefId};
use rustc_hir::{self as hir, def::Res, intravisit, QPath};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::{
    lint::in_external_macro,
    ty::{self, Ty},
};
use rustc_span::{sym, Span};

use clippy_utils::attrs::is_proc_macro;
use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::is_must_use_ty;
use clippy_utils::{match_def_path, must_use_attr, return_ty, trait_ref_of_method};

use super::{DOUBLE_MUST_USE, MUST_USE_CANDIDATE, MUST_USE_UNIT};

pub(super) fn check_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
    let attrs = cx.tcx.hir().attrs(item.hir_id());
    let attr = must_use_attr(attrs);
    if let hir::ItemKind::Fn(ref sig, ref _generics, ref body_id) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.hir_id(), item.span, fn_header_span, attr);
        } else if is_public && !is_proc_macro(cx.sess(), attrs) && !attrs.iter().any(|a| a.has_name(sym::no_mangle)) {
            check_must_use_candidate(
                cx,
                sig.decl,
                cx.tcx.hir().body(*body_id),
                item.span,
                item.def_id,
                item.span.with_hi(sig.decl.output.span().hi()),
                "this function could have a `#[must_use]` attribute",
            );
        }
    }
}

pub(super) fn check_impl_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
    if let hir::ImplItemKind::Fn(ref sig, ref body_id) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let attr = must_use_attr(attrs);
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.hir_id(), item.span, fn_header_span, attr);
        } else if is_public && !is_proc_macro(cx.sess(), attrs) && trait_ref_of_method(cx, item.def_id).is_none() {
            check_must_use_candidate(
                cx,
                sig.decl,
                cx.tcx.hir().body(*body_id),
                item.span,
                item.def_id,
                item.span.with_hi(sig.decl.output.span().hi()),
                "this method could have a `#[must_use]` attribute",
            );
        }
    }
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
    if let hir::TraitItemKind::Fn(ref sig, ref eid) = item.kind {
        let is_public = cx.access_levels.is_exported(item.def_id);
        let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());

        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let attr = must_use_attr(attrs);
        if let Some(attr) = attr {
            check_needless_must_use(cx, sig.decl, item.hir_id(), item.span, fn_header_span, attr);
        } else if let hir::TraitFn::Provided(eid) = *eid {
            let body = cx.tcx.hir().body(eid);
            if attr.is_none() && is_public && !is_proc_macro(cx.sess(), attrs) {
                check_must_use_candidate(
                    cx,
                    sig.decl,
                    body,
                    item.span,
                    item.def_id,
                    item.span.with_hi(sig.decl.output.span().hi()),
                    "this method could have a `#[must_use]` attribute",
                );
            }
        }
    }
}

fn check_needless_must_use(
    cx: &LateContext<'_>,
    decl: &hir::FnDecl<'_>,
    item_id: hir::HirId,
    item_span: Span,
    fn_header_span: Span,
    attr: &Attribute,
) {
    if in_external_macro(cx.sess(), item_span) {
        return;
    }
    if returns_unit(decl) {
        span_lint_and_then(
            cx,
            MUST_USE_UNIT,
            fn_header_span,
            "this unit-returning function has a `#[must_use]` attribute",
            |diag| {
                diag.span_suggestion(
                    attr.span,
                    "remove the attribute",
                    "".into(),
                    Applicability::MachineApplicable,
                );
            },
        );
    } else if attr.value_str().is_none() && is_must_use_ty(cx, return_ty(cx, item_id)) {
        span_lint_and_help(
            cx,
            DOUBLE_MUST_USE,
            fn_header_span,
            "this function has an empty `#[must_use]` attribute, but returns a type already marked as `#[must_use]`",
            None,
            "either add some descriptive text or remove the attribute",
        );
    }
}

fn check_must_use_candidate<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &'tcx hir::FnDecl<'_>,
    body: &'tcx hir::Body<'_>,
    item_span: Span,
    item_id: LocalDefId,
    fn_span: Span,
    msg: &str,
) {
    if has_mutable_arg(cx, body)
        || mutates_static(cx, body)
        || in_external_macro(cx.sess(), item_span)
        || returns_unit(decl)
        || !cx.access_levels.is_exported(item_id)
        || is_must_use_ty(cx, return_ty(cx, cx.tcx.hir().local_def_id_to_hir_id(item_id)))
    {
        return;
    }
    span_lint_and_then(cx, MUST_USE_CANDIDATE, fn_span, msg, |diag| {
        if let Some(snippet) = snippet_opt(cx, fn_span) {
            diag.span_suggestion(
                fn_span,
                "add the attribute",
                format!("#[must_use] {}", snippet),
                Applicability::MachineApplicable,
            );
        }
    });
}

fn returns_unit(decl: &hir::FnDecl<'_>) -> bool {
    match decl.output {
        hir::FnRetTy::DefaultReturn(_) => true,
        hir::FnRetTy::Return(ty) => match ty.kind {
            hir::TyKind::Tup(tys) => tys.is_empty(),
            hir::TyKind::Never => true,
            _ => false,
        },
    }
}

fn has_mutable_arg(cx: &LateContext<'_>, body: &hir::Body<'_>) -> bool {
    let mut tys = DefIdSet::default();
    body.params.iter().any(|param| is_mutable_pat(cx, param.pat, &mut tys))
}

fn is_mutable_pat(cx: &LateContext<'_>, pat: &hir::Pat<'_>, tys: &mut DefIdSet) -> bool {
    if let hir::PatKind::Wild = pat.kind {
        return false; // ignore `_` patterns
    }
    if cx.tcx.has_typeck_results(pat.hir_id.owner.to_def_id()) {
        is_mutable_ty(cx, cx.tcx.typeck(pat.hir_id.owner).pat_ty(pat), pat.span, tys)
    } else {
        false
    }
}

static KNOWN_WRAPPER_TYS: &[&[&str]] = &[&["alloc", "rc", "Rc"], &["std", "sync", "Arc"]];

fn is_mutable_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, span: Span, tys: &mut DefIdSet) -> bool {
    match *ty.kind() {
        // primitive types are never mutable
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str => false,
        ty::Adt(adt, substs) => {
            tys.insert(adt.did) && !ty.is_freeze(cx.tcx.at(span), cx.param_env)
                || KNOWN_WRAPPER_TYS.iter().any(|path| match_def_path(cx, adt.did, path))
                    && substs.types().any(|ty| is_mutable_ty(cx, ty, span, tys))
        },
        ty::Tuple(substs) => substs.types().any(|ty| is_mutable_ty(cx, ty, span, tys)),
        ty::Array(ty, _) | ty::Slice(ty) => is_mutable_ty(cx, ty, span, tys),
        ty::RawPtr(ty::TypeAndMut { ty, mutbl }) | ty::Ref(_, ty, mutbl) => {
            mutbl == hir::Mutability::Mut || is_mutable_ty(cx, ty, span, tys)
        },
        // calling something constitutes a side effect, so return true on all callables
        // also never calls need not be used, so return true for them, too
        _ => true,
    }
}

struct StaticMutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    mutates_static: bool,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for StaticMutVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        use hir::ExprKind::{AddrOf, Assign, AssignOp, Call, MethodCall};

        if self.mutates_static {
            return;
        }
        match expr.kind {
            Call(_, args) | MethodCall(_, args, _) => {
                let mut tys = DefIdSet::default();
                for arg in args {
                    if self.cx.tcx.has_typeck_results(arg.hir_id.owner.to_def_id())
                        && is_mutable_ty(
                            self.cx,
                            self.cx.tcx.typeck(arg.hir_id.owner).expr_ty(arg),
                            arg.span,
                            &mut tys,
                        )
                        && is_mutated_static(arg)
                    {
                        self.mutates_static = true;
                        return;
                    }
                    tys.clear();
                }
            },
            Assign(target, ..) | AssignOp(_, target, _) | AddrOf(_, hir::Mutability::Mut, target) => {
                self.mutates_static |= is_mutated_static(target);
            },
            _ => {},
        }
    }
}

fn is_mutated_static(e: &hir::Expr<'_>) -> bool {
    use hir::ExprKind::{Field, Index, Path};

    match e.kind {
        Path(QPath::Resolved(_, path)) => !matches!(path.res, Res::Local(_)),
        Path(_) => true,
        Field(inner, _) | Index(inner, _) => is_mutated_static(inner),
        _ => false,
    }
}

fn mutates_static<'tcx>(cx: &LateContext<'tcx>, body: &'tcx hir::Body<'_>) -> bool {
    let mut v = StaticMutVisitor {
        cx,
        mutates_static: false,
    };
    intravisit::walk_expr(&mut v, &body.value);
    v.mutates_static
}
