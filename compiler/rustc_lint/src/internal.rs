//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use rustc_hir::HirId;
use rustc_hir::def::Res;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericArgsRef, Ty as MiddleTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::{Span, sym};
use tracing::debug;
use {rustc_ast as ast, rustc_hir as hir};

use crate::lints::{
    BadOptAccessDiag, DefaultHashTypesDiag, DiagOutOfImpl, LintPassByHand,
    NonGlobImportTypeIrInherent, QueryInstability, QueryUntracked, SpanUseEqCtxtDiag,
    SymbolInternStringLiteralDiag, TyQualified, TykindDiag, TykindKind, TypeIrInherentUsage,
    TypeIrTraitUsage, UntranslatableDiag,
};
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};

declare_tool_lint! {
    /// The `default_hash_type` lint detects use of [`std::collections::HashMap`] and
    /// [`std::collections::HashSet`], suggesting the use of `FxHashMap`/`FxHashSet`.
    ///
    /// This can help as `FxHasher` can perform better than the default hasher. DOS protection is
    /// not required as input is assumed to be trusted.
    pub rustc::DEFAULT_HASH_TYPES,
    Allow,
    "forbid HashMap and HashSet and suggest the FxHash* variants",
    report_in_external_macro: true
}

declare_lint_pass!(DefaultHashTypes => [DEFAULT_HASH_TYPES]);

impl LateLintPass<'_> for DefaultHashTypes {
    fn check_path(&mut self, cx: &LateContext<'_>, path: &hir::Path<'_>, hir_id: HirId) {
        let Res::Def(rustc_hir::def::DefKind::Struct, def_id) = path.res else { return };
        if matches!(
            cx.tcx.hir_node(hir_id),
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Use(..), .. })
        ) {
            // Don't lint imports, only actual usages.
            return;
        }
        let preferred = match cx.tcx.get_diagnostic_name(def_id) {
            Some(sym::HashMap) => "FxHashMap",
            Some(sym::HashSet) => "FxHashSet",
            _ => return,
        };
        cx.emit_span_lint(
            DEFAULT_HASH_TYPES,
            path.span,
            DefaultHashTypesDiag { preferred, used: cx.tcx.item_name(def_id) },
        );
    }
}

/// Helper function for lints that check for expressions with calls and use typeck results to
/// get the `DefId` and `GenericArgsRef` of the function.
fn typeck_results_of_method_fn<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &hir::Expr<'_>,
) -> Option<(Span, DefId, ty::GenericArgsRef<'tcx>)> {
    match expr.kind {
        hir::ExprKind::MethodCall(segment, ..)
            if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) =>
        {
            Some((segment.ident.span, def_id, cx.typeck_results().node_args(expr.hir_id)))
        }
        _ => match cx.typeck_results().node_type(expr.hir_id).kind() {
            &ty::FnDef(def_id, args) => Some((expr.span, def_id, args)),
            _ => None,
        },
    }
}

declare_tool_lint! {
    /// The `potential_query_instability` lint detects use of methods which can lead to
    /// potential query instability, such as iterating over a `HashMap`.
    ///
    /// Due to the [incremental compilation](https://rustc-dev-guide.rust-lang.org/queries/incremental-compilation.html) model,
    /// queries must return deterministic, stable results. `HashMap` iteration order can change
    /// between compilations, and will introduce instability if query results expose the order.
    pub rustc::POTENTIAL_QUERY_INSTABILITY,
    Allow,
    "require explicit opt-in when using potentially unstable methods or functions",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// The `untracked_query_information` lint detects use of methods which leak information not
    /// tracked by the query system, such as whether a `Steal<T>` value has already been stolen. In
    /// order not to break incremental compilation, such methods must be used very carefully or not
    /// at all.
    pub rustc::UNTRACKED_QUERY_INFORMATION,
    Allow,
    "require explicit opt-in when accessing information not tracked by the query system",
    report_in_external_macro: true
}

declare_lint_pass!(QueryStability => [POTENTIAL_QUERY_INSTABILITY, UNTRACKED_QUERY_INFORMATION]);

impl LateLintPass<'_> for QueryStability {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let Some((span, def_id, args)) = typeck_results_of_method_fn(cx, expr) else { return };
        if let Ok(Some(instance)) = ty::Instance::try_resolve(cx.tcx, cx.typing_env(), def_id, args)
        {
            let def_id = instance.def_id();
            if cx.tcx.has_attr(def_id, sym::rustc_lint_query_instability) {
                cx.emit_span_lint(
                    POTENTIAL_QUERY_INSTABILITY,
                    span,
                    QueryInstability { query: cx.tcx.item_name(def_id) },
                );
            }
            if cx.tcx.has_attr(def_id, sym::rustc_lint_untracked_query_information) {
                cx.emit_span_lint(
                    UNTRACKED_QUERY_INFORMATION,
                    span,
                    QueryUntracked { method: cx.tcx.item_name(def_id) },
                );
            }
        }
    }
}

declare_tool_lint! {
    /// The `usage_of_ty_tykind` lint detects usages of `ty::TyKind::<kind>`,
    /// where `ty::<kind>` would suffice.
    pub rustc::USAGE_OF_TY_TYKIND,
    Allow,
    "usage of `ty::TyKind` outside of the `ty::sty` module",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// The `usage_of_qualified_ty` lint detects usages of `ty::TyKind`,
    /// where `Ty` should be used instead.
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
            let span =
                path.span.with_hi(segment.args.map_or(segment.ident.span, |a| a.span_ext).hi());
            cx.emit_span_lint(USAGE_OF_TY_TYKIND, path.span, TykindKind { suggestion: span });
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) {
        match &ty.kind {
            hir::TyKind::Path(hir::QPath::Resolved(_, path)) => {
                if lint_ty_kind_usage(cx, &path.res) {
                    let span = match cx.tcx.parent_hir_node(ty.hir_id) {
                        hir::Node::PatExpr(hir::PatExpr {
                            kind: hir::PatExprKind::Path(qpath),
                            ..
                        })
                        | hir::Node::Pat(hir::Pat {
                            kind:
                                hir::PatKind::TupleStruct(qpath, ..) | hir::PatKind::Struct(qpath, ..),
                            ..
                        })
                        | hir::Node::Expr(
                            hir::Expr { kind: hir::ExprKind::Path(qpath), .. }
                            | &hir::Expr { kind: hir::ExprKind::Struct(qpath, ..), .. },
                        ) => {
                            if let hir::QPath::TypeRelative(qpath_ty, ..) = qpath
                                && qpath_ty.hir_id == ty.hir_id
                            {
                                Some(path.span)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    match span {
                        Some(span) => {
                            cx.emit_span_lint(
                                USAGE_OF_TY_TYKIND,
                                path.span,
                                TykindKind { suggestion: span },
                            );
                        }
                        None => cx.emit_span_lint(USAGE_OF_TY_TYKIND, path.span, TykindDiag),
                    }
                } else if !ty.span.from_expansion()
                    && path.segments.len() > 1
                    && let Some(ty) = is_ty_or_ty_ctxt(cx, path)
                {
                    cx.emit_span_lint(
                        USAGE_OF_QUALIFIED_TY,
                        path.span,
                        TyQualified { ty, suggestion: path.span },
                    );
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

fn is_ty_or_ty_ctxt(cx: &LateContext<'_>, path: &hir::Path<'_>) -> Option<String> {
    match &path.res {
        Res::Def(_, def_id) => {
            if let Some(name @ (sym::Ty | sym::TyCtxt)) = cx.tcx.get_diagnostic_name(*def_id) {
                return Some(format!("{}{}", name, gen_args(path.segments.last().unwrap())));
            }
        }
        // Only lint on `&Ty` and `&TyCtxt` if it is used outside of a trait.
        Res::SelfTyAlias { alias_to: did, is_trait_impl: false, .. } => {
            if let ty::Adt(adt, args) = cx.tcx.type_of(did).instantiate_identity().kind()
                && let Some(name @ (sym::Ty | sym::TyCtxt)) = cx.tcx.get_diagnostic_name(adt.did())
            {
                return Some(format!("{}<{}>", name, args[0]));
            }
        }
        _ => (),
    }

    None
}

fn gen_args(segment: &hir::PathSegment<'_>) -> String {
    if let Some(args) = &segment.args {
        let lifetimes = args
            .args
            .iter()
            .filter_map(|arg| {
                if let hir::GenericArg::Lifetime(lt) = arg {
                    Some(lt.ident.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if !lifetimes.is_empty() {
            return format!("<{}>", lifetimes.join(", "));
        }
    }

    String::new()
}

declare_tool_lint! {
    /// The `non_glob_import_of_type_ir_inherent_item` lint detects
    /// non-glob imports of module `rustc_type_ir::inherent`.
    pub rustc::NON_GLOB_IMPORT_OF_TYPE_IR_INHERENT,
    Allow,
    "non-glob import of `rustc_type_ir::inherent`",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// The `usage_of_type_ir_inherent` lint detects usage of `rustc_type_ir::inherent`.
    ///
    /// This module should only be used within the trait solver.
    pub rustc::USAGE_OF_TYPE_IR_INHERENT,
    Allow,
    "usage `rustc_type_ir::inherent` outside of trait system",
    report_in_external_macro: true
}

declare_tool_lint! {
    /// The `usage_of_type_ir_traits` lint detects usage of `rustc_type_ir::Interner`,
    /// or `rustc_infer::InferCtxtLike`.
    ///
    /// Methods of this trait should only be used within the type system abstraction layer,
    /// and in the generic next trait solver implementation. Look for an analogously named
    /// method on `TyCtxt` or `InferCtxt` (respectively).
    pub rustc::USAGE_OF_TYPE_IR_TRAITS,
    Allow,
    "usage `rustc_type_ir`-specific abstraction traits outside of trait system",
    report_in_external_macro: true
}

declare_lint_pass!(TypeIr => [NON_GLOB_IMPORT_OF_TYPE_IR_INHERENT, USAGE_OF_TYPE_IR_INHERENT, USAGE_OF_TYPE_IR_TRAITS]);

impl<'tcx> LateLintPass<'tcx> for TypeIr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        let res_def_id = match expr.kind {
            hir::ExprKind::Path(hir::QPath::Resolved(_, path)) => path.res.opt_def_id(),
            hir::ExprKind::Path(hir::QPath::TypeRelative(..)) | hir::ExprKind::MethodCall(..) => {
                cx.typeck_results().type_dependent_def_id(expr.hir_id)
            }
            _ => return,
        };
        let Some(res_def_id) = res_def_id else {
            return;
        };
        if let Some(assoc_item) = cx.tcx.opt_associated_item(res_def_id)
            && let Some(trait_def_id) = assoc_item.trait_container(cx.tcx)
            && (cx.tcx.is_diagnostic_item(sym::type_ir_interner, trait_def_id)
                | cx.tcx.is_diagnostic_item(sym::type_ir_infer_ctxt_like, trait_def_id))
        {
            cx.emit_span_lint(USAGE_OF_TYPE_IR_TRAITS, expr.span, TypeIrTraitUsage);
        }
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'tcx>) {
        let rustc_hir::ItemKind::Use(path, kind) = item.kind else { return };

        let is_mod_inherent = |res: Res| {
            res.opt_def_id()
                .is_some_and(|def_id| cx.tcx.is_diagnostic_item(sym::type_ir_inherent, def_id))
        };

        // Path segments except for the final.
        if let Some(seg) = path.segments.iter().find(|seg| is_mod_inherent(seg.res)) {
            cx.emit_span_lint(USAGE_OF_TYPE_IR_INHERENT, seg.ident.span, TypeIrInherentUsage);
        }
        // Final path resolutions, like `use rustc_type_ir::inherent`
        else if let Some(type_ns) = path.res.type_ns
            && is_mod_inherent(type_ns)
        {
            cx.emit_span_lint(
                USAGE_OF_TYPE_IR_INHERENT,
                path.segments.last().unwrap().ident.span,
                TypeIrInherentUsage,
            );
        }

        let (lo, hi, snippet) = match path.segments {
            [.., penultimate, segment] if is_mod_inherent(penultimate.res) => {
                (segment.ident.span, item.kind.ident().unwrap().span, "*")
            }
            [.., segment]
                if let Some(type_ns) = path.res.type_ns
                    && is_mod_inherent(type_ns)
                    && let rustc_hir::UseKind::Single(ident) = kind =>
            {
                let (lo, snippet) =
                    match cx.tcx.sess.source_map().span_to_snippet(path.span).as_deref() {
                        Ok("self") => (path.span, "*"),
                        _ => (segment.ident.span.shrink_to_hi(), "::*"),
                    };
                (lo, if segment.ident == ident { lo } else { ident.span }, snippet)
            }
            _ => return,
        };
        cx.emit_span_lint(
            NON_GLOB_IMPORT_OF_TYPE_IR_INHERENT,
            path.span,
            NonGlobImportTypeIrInherent { suggestion: lo.eq_ctxt(hi).then(|| lo.to(hi)), snippet },
        );
    }
}

declare_tool_lint! {
    /// The `lint_pass_impl_without_macro` detects manual implementations of a lint
    /// pass, without using [`declare_lint_pass`] or [`impl_lint_pass`].
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
                        cx.emit_span_lint(
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
    /// The `untranslatable_diagnostic` lint detects messages passed to functions with `impl
    /// Into<{D,Subd}iagMessage` parameters without using translatable Fluent strings.
    ///
    /// More details on translatable diagnostics can be found
    /// [here](https://rustc-dev-guide.rust-lang.org/diagnostics/translation.html).
    pub rustc::UNTRANSLATABLE_DIAGNOSTIC,
    Allow,
    "prevent creation of diagnostics which cannot be translated",
    report_in_external_macro: true,
    @eval_always = true
}

declare_tool_lint! {
    /// The `diagnostic_outside_of_impl` lint detects calls to functions annotated with
    /// `#[rustc_lint_diagnostics]` that are outside an `Diagnostic`, `Subdiagnostic`, or
    /// `LintDiagnostic` impl (either hand-written or derived).
    ///
    /// More details on diagnostics implementations can be found
    /// [here](https://rustc-dev-guide.rust-lang.org/diagnostics/diagnostic-structs.html).
    pub rustc::DIAGNOSTIC_OUTSIDE_OF_IMPL,
    Allow,
    "prevent diagnostic creation outside of `Diagnostic`/`Subdiagnostic`/`LintDiagnostic` impls",
    report_in_external_macro: true,
    @eval_always = true
}

declare_lint_pass!(Diagnostics => [UNTRANSLATABLE_DIAGNOSTIC, DIAGNOSTIC_OUTSIDE_OF_IMPL]);

impl LateLintPass<'_> for Diagnostics {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let collect_args_tys_and_spans = |args: &[hir::Expr<'_>], reserve_one_extra: bool| {
            let mut result = Vec::with_capacity(args.len() + usize::from(reserve_one_extra));
            result.extend(args.iter().map(|arg| (cx.typeck_results().expr_ty(arg), arg.span)));
            result
        };
        // Only check function calls and method calls.
        let (span, def_id, fn_gen_args, arg_tys_and_spans) = match expr.kind {
            hir::ExprKind::Call(callee, args) => {
                match cx.typeck_results().node_type(callee.hir_id).kind() {
                    &ty::FnDef(def_id, fn_gen_args) => {
                        (callee.span, def_id, fn_gen_args, collect_args_tys_and_spans(args, false))
                    }
                    _ => return, // occurs for fns passed as args
                }
            }
            hir::ExprKind::MethodCall(_segment, _recv, args, _span) => {
                let Some((span, def_id, fn_gen_args)) = typeck_results_of_method_fn(cx, expr)
                else {
                    return;
                };
                let mut args = collect_args_tys_and_spans(args, true);
                args.insert(0, (cx.tcx.types.self_param, _recv.span)); // dummy inserted for `self`
                (span, def_id, fn_gen_args, args)
            }
            _ => return,
        };

        Self::diagnostic_outside_of_impl(cx, span, expr.hir_id, def_id, fn_gen_args);
        Self::untranslatable_diagnostic(cx, def_id, &arg_tys_and_spans);
    }
}

impl Diagnostics {
    // Is the type `{D,Subd}iagMessage`?
    fn is_diag_message<'cx>(cx: &LateContext<'cx>, ty: MiddleTy<'cx>) -> bool {
        if let Some(adt_def) = ty.ty_adt_def()
            && let Some(name) = cx.tcx.get_diagnostic_name(adt_def.did())
            && matches!(name, sym::DiagMessage | sym::SubdiagMessage)
        {
            true
        } else {
            false
        }
    }

    fn untranslatable_diagnostic<'cx>(
        cx: &LateContext<'cx>,
        def_id: DefId,
        arg_tys_and_spans: &[(MiddleTy<'cx>, Span)],
    ) {
        let fn_sig = cx.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        let predicates = cx.tcx.predicates_of(def_id).instantiate_identity(cx.tcx).predicates;
        for (i, &param_ty) in fn_sig.inputs().iter().enumerate() {
            if let ty::Param(sig_param) = param_ty.kind() {
                // It is a type parameter. Check if it is `impl Into<{D,Subd}iagMessage>`.
                for pred in predicates.iter() {
                    if let Some(trait_pred) = pred.as_trait_clause()
                        && let trait_ref = trait_pred.skip_binder().trait_ref
                        && trait_ref.self_ty() == param_ty // correct predicate for the param?
                        && cx.tcx.is_diagnostic_item(sym::Into, trait_ref.def_id)
                        && let ty1 = trait_ref.args.type_at(1)
                        && Self::is_diag_message(cx, ty1)
                    {
                        // Calls to methods with an `impl Into<{D,Subd}iagMessage>` parameter must be passed an arg
                        // with type `{D,Subd}iagMessage` or `impl Into<{D,Subd}iagMessage>`. Otherwise, emit an
                        // `UNTRANSLATABLE_DIAGNOSTIC` lint.
                        let (arg_ty, arg_span) = arg_tys_and_spans[i];

                        // Is the arg type `{Sub,D}iagMessage`or `impl Into<{Sub,D}iagMessage>`?
                        let is_translatable = Self::is_diag_message(cx, arg_ty)
                            || matches!(arg_ty.kind(), ty::Param(arg_param) if arg_param.name == sig_param.name);
                        if !is_translatable {
                            cx.emit_span_lint(
                                UNTRANSLATABLE_DIAGNOSTIC,
                                arg_span,
                                UntranslatableDiag,
                            );
                        }
                    }
                }
            }
        }
    }

    fn diagnostic_outside_of_impl<'cx>(
        cx: &LateContext<'cx>,
        span: Span,
        current_id: HirId,
        def_id: DefId,
        fn_gen_args: GenericArgsRef<'cx>,
    ) {
        // Is the callee marked with `#[rustc_lint_diagnostics]`?
        let Some(inst) =
            ty::Instance::try_resolve(cx.tcx, cx.typing_env(), def_id, fn_gen_args).ok().flatten()
        else {
            return;
        };
        let has_attr = cx.tcx.has_attr(inst.def_id(), sym::rustc_lint_diagnostics);
        if !has_attr {
            return;
        };

        for (hir_id, _parent) in cx.tcx.hir_parent_iter(current_id) {
            if let Some(owner_did) = hir_id.as_owner()
                && cx.tcx.has_attr(owner_did, sym::rustc_lint_diagnostics)
            {
                // The parent method is marked with `#[rustc_lint_diagnostics]`
                return;
            }
        }

        // Calls to `#[rustc_lint_diagnostics]`-marked functions should only occur:
        // - inside an impl of `Diagnostic`, `Subdiagnostic`, or `LintDiagnostic`, or
        // - inside a parent function that is itself marked with `#[rustc_lint_diagnostics]`.
        //
        // Otherwise, emit a `DIAGNOSTIC_OUTSIDE_OF_IMPL` lint.
        let mut is_inside_appropriate_impl = false;
        for (_hir_id, parent) in cx.tcx.hir_parent_iter(current_id) {
            debug!(?parent);
            if let hir::Node::Item(hir::Item { kind: hir::ItemKind::Impl(impl_), .. }) = parent
                && let hir::Impl { of_trait: Some(of_trait), .. } = impl_
                && let Some(def_id) = of_trait.trait_def_id()
                && let Some(name) = cx.tcx.get_diagnostic_name(def_id)
                && matches!(name, sym::Diagnostic | sym::Subdiagnostic | sym::LintDiagnostic)
            {
                is_inside_appropriate_impl = true;
                break;
            }
        }
        debug!(?is_inside_appropriate_impl);
        if !is_inside_appropriate_impl {
            cx.emit_span_lint(DIAGNOSTIC_OUTSIDE_OF_IMPL, span, DiagOutOfImpl);
        }
    }
}

declare_tool_lint! {
    /// The `bad_opt_access` lint detects accessing options by field instead of
    /// the wrapper function.
    pub rustc::BAD_OPT_ACCESS,
    Deny,
    "prevent using options by field access when there is a wrapper function",
    report_in_external_macro: true
}

declare_lint_pass!(BadOptAccess => [BAD_OPT_ACCESS]);

impl LateLintPass<'_> for BadOptAccess {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &hir::Expr<'_>) {
        let hir::ExprKind::Field(base, target) = expr.kind else { return };
        let Some(adt_def) = cx.typeck_results().expr_ty(base).ty_adt_def() else { return };
        // Skip types without `#[rustc_lint_opt_ty]` - only so that the rest of the lint can be
        // avoided.
        if !cx.tcx.has_attr(adt_def.did(), sym::rustc_lint_opt_ty) {
            return;
        }

        for field in adt_def.all_fields() {
            if field.name == target.name
                && let Some(attr) =
                    cx.tcx.get_attr(field.did, sym::rustc_lint_opt_deny_field_access)
                && let Some(items) = attr.meta_item_list()
                && let Some(item) = items.first()
                && let Some(lit) = item.lit()
                && let ast::LitKind::Str(val, _) = lit.kind
            {
                cx.emit_span_lint(
                    BAD_OPT_ACCESS,
                    expr.span,
                    BadOptAccessDiag { msg: val.as_str() },
                );
            }
        }
    }
}

declare_tool_lint! {
    pub rustc::SPAN_USE_EQ_CTXT,
    Allow,
    "forbid uses of `==` with `Span::ctxt`, suggest `Span::eq_ctxt` instead",
    report_in_external_macro: true
}

declare_lint_pass!(SpanUseEqCtxt => [SPAN_USE_EQ_CTXT]);

impl<'tcx> LateLintPass<'tcx> for SpanUseEqCtxt {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &hir::Expr<'_>) {
        if let hir::ExprKind::Binary(
            hir::BinOp { node: hir::BinOpKind::Eq | hir::BinOpKind::Ne, .. },
            lhs,
            rhs,
        ) = expr.kind
        {
            if is_span_ctxt_call(cx, lhs) && is_span_ctxt_call(cx, rhs) {
                cx.emit_span_lint(SPAN_USE_EQ_CTXT, expr.span, SpanUseEqCtxtDiag);
            }
        }
    }
}

fn is_span_ctxt_call(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    match &expr.kind {
        hir::ExprKind::MethodCall(..) => cx
            .typeck_results()
            .type_dependent_def_id(expr.hir_id)
            .is_some_and(|call_did| cx.tcx.is_diagnostic_item(sym::SpanCtxt, call_did)),

        _ => false,
    }
}

declare_tool_lint! {
    /// The `symbol_intern_string_literal` detects `Symbol::intern` being called on a string literal
    pub rustc::SYMBOL_INTERN_STRING_LITERAL,
    // rustc_driver crates out of the compiler can't/shouldn't add preinterned symbols;
    // bootstrap will deny this manually
    Allow,
    "Forbid uses of string literals in `Symbol::intern`, suggesting preinterning instead",
    report_in_external_macro: true
}

declare_lint_pass!(SymbolInternStringLiteral => [SYMBOL_INTERN_STRING_LITERAL]);

impl<'tcx> LateLintPass<'tcx> for SymbolInternStringLiteral {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx rustc_hir::Expr<'tcx>) {
        if let hir::ExprKind::Call(path, [arg]) = expr.kind
            && let hir::ExprKind::Path(ref qpath) = path.kind
            && let Some(def_id) = cx.qpath_res(qpath, path.hir_id).opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::SymbolIntern, def_id)
            && let hir::ExprKind::Lit(kind) = arg.kind
            && let rustc_ast::LitKind::Str(_, _) = kind.node
        {
            cx.emit_span_lint(
                SYMBOL_INTERN_STRING_LITERAL,
                kind.span,
                SymbolInternStringLiteralDiag,
            );
        }
    }
}
