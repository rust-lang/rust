//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use crate::lints::{
    BadOptAccessDiag, DefaultHashTypesDiag, DiagOutOfImpl, LintPassByHand, NonExistantDocKeyword,
    QueryInstability, TyQualified, TykindDiag, TykindKind, UntranslatableDiag,
};
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_hir::def::Res;
use rustc_hir::{def_id::DefId, Expr, ExprKind, GenericArg, PatKind, Path, PathSegment, QPath};
use rustc_hir::{HirId, Impl, Item, ItemKind, Node, Pat, Ty, TyKind};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

declare_tool_lint! {
    pub rustc::DEFAULT_HASH_TYPES,
    Allow,
    "forbid HashMap and HashSet and suggest the FxHash* variants",
    report_in_external_macro: true
}

declare_lint_pass!(DefaultHashTypes => [DEFAULT_HASH_TYPES]);

impl LateLintPass<'_> for DefaultHashTypes {
    fn check_path(&mut self, cx: &LateContext<'_>, path: &Path<'_>, hir_id: HirId) {
        let Res::Def(rustc_hir::def::DefKind::Struct, def_id) = path.res else { return };
        if matches!(cx.tcx.hir().get(hir_id), Node::Item(Item { kind: ItemKind::Use(..), .. })) {
            // don't lint imports, only actual usages
            return;
        }
        let preferred = match cx.tcx.get_diagnostic_name(def_id) {
            Some(sym::HashMap) => "FxHashMap",
            Some(sym::HashSet) => "FxHashSet",
            _ => return,
        };
        cx.emit_spanned_lint(
            DEFAULT_HASH_TYPES,
            path.span,
            DefaultHashTypesDiag { preferred, used: cx.tcx.item_name(def_id) },
        );
    }
}

/// Helper function for lints that check for expressions with calls and use typeck results to
/// get the `DefId` and `SubstsRef` of the function.
fn typeck_results_of_method_fn<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &Expr<'_>,
) -> Option<(Span, DefId, ty::subst::SubstsRef<'tcx>)> {
    match expr.kind {
        ExprKind::MethodCall(segment, ..)
            if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) =>
        {
            Some((segment.ident.span, def_id, cx.typeck_results().node_substs(expr.hir_id)))
        },
        _ => {
            match cx.typeck_results().node_type(expr.hir_id).kind() {
                &ty::FnDef(def_id, substs) => Some((expr.span, def_id, substs)),
                _ => None,
            }
        }
    }
}

declare_tool_lint! {
    pub rustc::POTENTIAL_QUERY_INSTABILITY,
    Allow,
    "require explicit opt-in when using potentially unstable methods or functions",
    report_in_external_macro: true
}

declare_lint_pass!(QueryStability => [POTENTIAL_QUERY_INSTABILITY]);

impl LateLintPass<'_> for QueryStability {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some((span, def_id, substs)) = typeck_results_of_method_fn(cx, expr) else { return };
        if let Ok(Some(instance)) = ty::Instance::resolve(cx.tcx, cx.param_env, def_id, substs) {
            let def_id = instance.def_id();
            if cx.tcx.has_attr(def_id, sym::rustc_lint_query_instability) {
                cx.emit_spanned_lint(
                    POTENTIAL_QUERY_INSTABILITY,
                    span,
                    QueryInstability { query: cx.tcx.item_name(def_id) },
                );
            }
        }
    }
}

declare_tool_lint! {
    pub rustc::USAGE_OF_TY_TYKIND,
    Allow,
    "usage of `ty::TyKind` outside of the `ty::sty` module",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub rustc::USAGE_OF_QUALIFIED_TY,
    Allow,
    "using `ty::{Ty,TyCtxt}` instead of importing it",
    report_in_external_macro: true
}

declare_lint_pass!(TyTyKind => [
    USAGE_OF_TY_TYKIND,
    USAGE_OF_QUALIFIED_TY,
]);

impl<'tcx> LateLintPass<'tcx> for TyTyKind {
    fn check_path(
        &mut self,
        cx: &LateContext<'tcx>,
        path: &rustc_hir::Path<'tcx>,
        _: rustc_hir::HirId,
    ) {
        if let Some(segment) = path.segments.iter().nth_back(1)
        && lint_ty_kind_usage(cx, &segment.res)
        {
            let span = path.span.with_hi(
                segment.args.map_or(segment.ident.span, |a| a.span_ext).hi()
            );
            cx.emit_spanned_lint(USAGE_OF_TY_TYKIND, path.span, TykindKind {
                suggestion: span,
            });
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx Ty<'tcx>) {
        match &ty.kind {
            TyKind::Path(QPath::Resolved(_, path)) => {
                if lint_ty_kind_usage(cx, &path.res) {
                    let hir = cx.tcx.hir();
                    let span = match hir.find_parent(ty.hir_id) {
                        Some(Node::Pat(Pat {
                            kind:
                                PatKind::Path(qpath)
                                | PatKind::TupleStruct(qpath, ..)
                                | PatKind::Struct(qpath, ..),
                            ..
                        })) => {
                            if let QPath::TypeRelative(qpath_ty, ..) = qpath
                                && qpath_ty.hir_id == ty.hir_id
                            {
                                Some(path.span)
                            } else {
                                None
                            }
                        }
                        Some(Node::Expr(Expr {
                            kind: ExprKind::Path(qpath),
                            ..
                        })) => {
                            if let QPath::TypeRelative(qpath_ty, ..) = qpath
                                && qpath_ty.hir_id == ty.hir_id
                            {
                                Some(path.span)
                            } else {
                                None
                            }
                        }
                        // Can't unify these two branches because qpath below is `&&` and above is `&`
                        // and `A | B` paths don't play well together with adjustments, apparently.
                        Some(Node::Expr(Expr {
                            kind: ExprKind::Struct(qpath, ..),
                            ..
                        })) => {
                            if let QPath::TypeRelative(qpath_ty, ..) = qpath
                                && qpath_ty.hir_id == ty.hir_id
                            {
                                Some(path.span)
                            } else {
                                None
                            }
                        }
                        _ => None
                    };

                    match span {
                        Some(span) => {
                            cx.emit_spanned_lint(USAGE_OF_TY_TYKIND, path.span, TykindKind {
                                suggestion: span,
                            });
                        },
                        None => cx.emit_spanned_lint(USAGE_OF_TY_TYKIND, path.span, TykindDiag),
                    }
                } else if !ty.span.from_expansion() && path.segments.len() > 1 && let Some(ty) = is_ty_or_ty_ctxt(cx, &path) {
                    cx.emit_spanned_lint(USAGE_OF_QUALIFIED_TY, path.span, TyQualified {
                        ty,
                        suggestion: path.span,
                    });
                }
            }
            _ => {}
        }
    }
}

fn lint_ty_kind_usage(cx: &LateContext<'_>, res: &Res) -> bool {
    if let Some(did) = res.opt_def_id() {
        cx.tcx.is_diagnostic_item(sym::TyKind, did) || cx.tcx.is_diagnostic_item(sym::IrTyKind, did)
    } else {
        false
    }
}

fn is_ty_or_ty_ctxt(cx: &LateContext<'_>, path: &Path<'_>) -> Option<String> {
    match &path.res {
        Res::Def(_, def_id) => {
            if let Some(name @ (sym::Ty | sym::TyCtxt)) = cx.tcx.get_diagnostic_name(*def_id) {
                return Some(format!("{}{}", name, gen_args(path.segments.last().unwrap())));
            }
        }
        // Only lint on `&Ty` and `&TyCtxt` if it is used outside of a trait.
        Res::SelfTyAlias { alias_to: did, is_trait_impl: false, .. } => {
            if let ty::Adt(adt, substs) = cx.tcx.type_of(did).subst_identity().kind() {
                if let Some(name @ (sym::Ty | sym::TyCtxt)) = cx.tcx.get_diagnostic_name(adt.did())
                {
                    // NOTE: This path is currently unreachable as `Ty<'tcx>` is
                    // defined as a type alias meaning that `impl<'tcx> Ty<'tcx>`
                    // is not actually allowed.
                    //
                    // I(@lcnr) still kept this branch in so we don't miss this
                    // if we ever change it in the future.
                    return Some(format!("{}<{}>", name, substs[0]));
                }
            }
        }
        _ => (),
    }

    None
}

fn gen_args(segment: &PathSegment<'_>) -> String {
    if let Some(args) = &segment.args {
        let lifetimes = args
            .args
            .iter()
            .filter_map(|arg| {
                if let GenericArg::Lifetime(lt) = arg { Some(lt.ident.to_string()) } else { None }
            })
            .collect::<Vec<_>>();

        if !lifetimes.is_empty() {
            return format!("<{}>", lifetimes.join(", "));
        }
    }

    String::new()
}

declare_tool_lint! {
    pub rustc::LINT_PASS_IMPL_WITHOUT_MACRO,
    Allow,
    "`impl LintPass` without the `declare_lint_pass!` or `impl_lint_pass!` macros"
}

declare_lint_pass!(LintPassImpl => [LINT_PASS_IMPL_WITHOUT_MACRO]);

impl EarlyLintPass for LintPassImpl {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if let ast::ItemKind::Impl(box ast::Impl { of_trait: Some(lint_pass), .. }) = &item.kind {
            if let Some(last) = lint_pass.path.segments.last() {
                if last.ident.name == sym::LintPass {
                    let expn_data = lint_pass.path.span.ctxt().outer_expn_data();
                    let call_site = expn_data.call_site;
                    if expn_data.kind != ExpnKind::Macro(MacroKind::Bang, sym::impl_lint_pass)
                        && call_site.ctxt().outer_expn_data().kind
                            != ExpnKind::Macro(MacroKind::Bang, sym::declare_lint_pass)
                    {
                        cx.emit_spanned_lint(
                            LINT_PASS_IMPL_WITHOUT_MACRO,
                            lint_pass.path.span,
                            LintPassByHand,
                        );
                    }
                }
            }
        }
    }
}

declare_tool_lint! {
    pub rustc::EXISTING_DOC_KEYWORD,
    Allow,
    "Check that documented keywords in std and core actually exist",
    report_in_external_macro: true
}

declare_lint_pass!(ExistingDocKeyword => [EXISTING_DOC_KEYWORD]);

fn is_doc_keyword(s: Symbol) -> bool {
    s <= kw::Union
}

impl<'tcx> LateLintPass<'tcx> for ExistingDocKeyword {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &rustc_hir::Item<'_>) {
        for attr in cx.tcx.hir().attrs(item.hir_id()) {
            if !attr.has_name(sym::doc) {
                continue;
            }
            if let Some(list) = attr.meta_item_list() {
                for nested in list {
                    if nested.has_name(sym::keyword) {
                        let keyword = nested
                            .value_str()
                            .expect("#[doc(keyword = \"...\")] expected a value!");
                        if is_doc_keyword(keyword) {
                            return;
                        }
                        cx.emit_spanned_lint(
                            EXISTING_DOC_KEYWORD,
                            attr.span,
                            NonExistantDocKeyword { keyword },
                        );
                    }
                }
            }
        }
    }
}

declare_tool_lint! {
    pub rustc::UNTRANSLATABLE_DIAGNOSTIC,
    Allow,
    "prevent creation of diagnostics which cannot be translated",
    report_in_external_macro: true
}

declare_tool_lint! {
    pub rustc::DIAGNOSTIC_OUTSIDE_OF_IMPL,
    Allow,
    "prevent creation of diagnostics outside of `IntoDiagnostic`/`AddToDiagnostic` impls",
    report_in_external_macro: true
}

declare_lint_pass!(Diagnostics => [ UNTRANSLATABLE_DIAGNOSTIC, DIAGNOSTIC_OUTSIDE_OF_IMPL ]);

impl LateLintPass<'_> for Diagnostics {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let Some((span, def_id, substs)) = typeck_results_of_method_fn(cx, expr) else { return };
        debug!(?span, ?def_id, ?substs);
        let has_attr = ty::Instance::resolve(cx.tcx, cx.param_env, def_id, substs)
            .ok()
            .and_then(|inst| inst)
            .map(|inst| cx.tcx.has_attr(inst.def_id(), sym::rustc_lint_diagnostics))
            .unwrap_or(false);
        if !has_attr {
            return;
        }

        let mut found_parent_with_attr = false;
        let mut found_impl = false;
        for (hir_id, parent) in cx.tcx.hir().parent_iter(expr.hir_id) {
            if let Some(owner_did) = hir_id.as_owner() {
                found_parent_with_attr = found_parent_with_attr
                    || cx.tcx.has_attr(owner_did.to_def_id(), sym::rustc_lint_diagnostics);
            }

            debug!(?parent);
            if let Node::Item(Item { kind: ItemKind::Impl(impl_), .. }) = parent &&
                let Impl { of_trait: Some(of_trait), .. } = impl_ &&
                let Some(def_id) = of_trait.trait_def_id() &&
                let Some(name) = cx.tcx.get_diagnostic_name(def_id) &&
                matches!(name, sym::IntoDiagnostic | sym::AddToDiagnostic | sym::DecorateLint)
            {
                found_impl = true;
                break;
            }
        }
        debug!(?found_impl);
        if !found_parent_with_attr && !found_impl {
            cx.emit_spanned_lint(DIAGNOSTIC_OUTSIDE_OF_IMPL, span, DiagOutOfImpl);
        }

        let mut found_diagnostic_message = false;
        for ty in substs.types() {
            debug!(?ty);
            if let Some(adt_def) = ty.ty_adt_def() &&
                let Some(name) =  cx.tcx.get_diagnostic_name(adt_def.did()) &&
                matches!(name, sym::DiagnosticMessage | sym::SubdiagnosticMessage)
            {
                found_diagnostic_message = true;
                break;
            }
        }
        debug!(?found_diagnostic_message);
        if !found_parent_with_attr && !found_diagnostic_message {
            cx.emit_spanned_lint(UNTRANSLATABLE_DIAGNOSTIC, span, UntranslatableDiag);
        }
    }
}

declare_tool_lint! {
    pub rustc::BAD_OPT_ACCESS,
    Deny,
    "prevent using options by field access when there is a wrapper function",
    report_in_external_macro: true
}

declare_lint_pass!(BadOptAccess => [ BAD_OPT_ACCESS ]);

impl LateLintPass<'_> for BadOptAccess {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        let ExprKind::Field(base, target) = expr.kind else { return };
        let Some(adt_def) = cx.typeck_results().expr_ty(base).ty_adt_def() else { return };
        // Skip types without `#[rustc_lint_opt_ty]` - only so that the rest of the lint can be
        // avoided.
        if !cx.tcx.has_attr(adt_def.did(), sym::rustc_lint_opt_ty) {
            return;
        }

        for field in adt_def.all_fields() {
            if field.name == target.name &&
                let Some(attr) = cx.tcx.get_attr(field.did, sym::rustc_lint_opt_deny_field_access) &&
                let Some(items) = attr.meta_item_list()  &&
                let Some(item) = items.first()  &&
                let Some(lit) = item.lit()  &&
                let ast::LitKind::Str(val, _) = lit.kind
            {
                cx.emit_spanned_lint(BAD_OPT_ACCESS, expr.span, BadOptAccessDiag {
                    msg: val.as_str(),
                });
            }
        }
    }
}
