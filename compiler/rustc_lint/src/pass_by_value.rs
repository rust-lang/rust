use crate::lints::PassByValueDiag;
use crate::{LateContext, LateLintPass, LintContext};
use rustc_hir as hir;
use rustc_hir::def::Res;
use rustc_hir::{GenericArg, PathSegment, QPath, TyKind};
use rustc_middle::ty;
use rustc_span::symbol::sym;

declare_tool_lint! {
    /// The `rustc_pass_by_value` lint marks a type with `#[rustc_pass_by_value]` requiring it to
    /// always be passed by value. This is usually used for types that are thin wrappers around
    /// references, so there is no benefit to an extra layer of indirection. (Example: `Ty` which
    /// is a reference to an `Interned<TyKind>`)
    pub rustc::PASS_BY_VALUE,
    Warn,
    "pass by reference of a type flagged as `#[rustc_pass_by_value]`",
    report_in_external_macro: true
}

declare_lint_pass!(PassByValue => [PASS_BY_VALUE]);

impl<'tcx> LateLintPass<'tcx> for PassByValue {
    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx hir::Ty<'tcx>) {
        match &ty.kind {
            TyKind::Ref(_, hir::MutTy { ty: inner_ty, mutbl: hir::Mutability::Not }) => {
                if let Some(impl_did) = cx.tcx.impl_of_method(ty.hir_id.owner.to_def_id()) {
                    if cx.tcx.impl_trait_ref(impl_did).is_some() {
                        return;
                    }
                }
                if let Some(t) = path_for_pass_by_value(cx, &inner_ty) {
                    cx.emit_spanned_lint(
                        PASS_BY_VALUE,
                        ty.span,
                        PassByValueDiag { ty: t, suggestion: ty.span },
                    );
                }
            }
            _ => {}
        }
    }
}

fn path_for_pass_by_value(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> Option<String> {
    if let TyKind::Path(QPath::Resolved(_, path)) = &ty.kind {
        match path.res {
            Res::Def(_, def_id) if cx.tcx.has_attr(def_id, sym::rustc_pass_by_value) => {
                let name = cx.tcx.item_name(def_id).to_ident_string();
                let path_segment = path.segments.last().unwrap();
                return Some(format!("{}{}", name, gen_args(cx, path_segment)));
            }
            Res::SelfTyAlias { alias_to: did, is_trait_impl: false, .. } => {
                if let ty::Adt(adt, args) = cx.tcx.type_of(did).instantiate_identity().kind() {
                    if cx.tcx.has_attr(adt.did(), sym::rustc_pass_by_value) {
                        return Some(cx.tcx.def_path_str_with_args(adt.did(), args));
                    }
                }
            }
            _ => (),
        }
    }

    None
}

fn gen_args(cx: &LateContext<'_>, segment: &PathSegment<'_>) -> String {
    if let Some(args) = &segment.args {
        let params = args
            .args
            .iter()
            .map(|arg| match arg {
                GenericArg::Lifetime(lt) => lt.to_string(),
                GenericArg::Type(ty) => {
                    cx.tcx.sess.source_map().span_to_snippet(ty.span).unwrap_or_else(|_| "_".into())
                }
                GenericArg::Const(c) => {
                    cx.tcx.sess.source_map().span_to_snippet(c.span).unwrap_or_else(|_| "_".into())
                }
                GenericArg::Infer(_) => String::from("_"),
            })
            .collect::<Vec<_>>();

        if !params.is_empty() {
            return format!("<{}>", params.join(", "));
        }
    }

    String::new()
}
