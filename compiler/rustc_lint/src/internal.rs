//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{Expr, ExprKind, GenericArg, Path, PathSegment, QPath};
use rustc_hir::{HirId, Item, ItemKind, Node, Ty, TyKind};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::symbol::{kw, sym, Symbol};

declare_tool_lint! {
    pub rustc::DEFAULT_HASH_TYPES,
    Allow,
    "forbid HashMap and HashSet and suggest the FxHash* variants",
    report_in_external_macro: true
}

declare_lint_pass!(DefaultHashTypes => [DEFAULT_HASH_TYPES]);

impl LateLintPass<'_> for DefaultHashTypes {
    fn check_path(&mut self, cx: &LateContext<'_>, path: &Path<'_>, hir_id: HirId) {
        let def_id = match path.res {
            Res::Def(rustc_hir::def::DefKind::Struct, id) => id,
            _ => return,
        };
        if matches!(cx.tcx.hir().get(hir_id), Node::Item(Item { kind: ItemKind::Use(..), .. })) {
            // don't lint imports, only actual usages
            return;
        }
        let replace = match cx.tcx.get_diagnostic_name(def_id) {
            Some(sym::HashMap) => "FxHashMap",
            Some(sym::HashSet) => "FxHashSet",
            _ => return,
        };
        cx.struct_span_lint(DEFAULT_HASH_TYPES, path.span, |lint| {
            let msg = format!(
                "prefer `{}` over `{}`, it has better performance",
                replace,
                cx.tcx.item_name(def_id)
            );
            lint.build(&msg)
                .note(&format!("a `use rustc_data_structures::fx::{}` may be necessary", replace))
                .emit();
        });
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
        // FIXME(rustdoc): This lint uses typecheck results, causing rustdoc to
        // error if there are resolution failures.
        //
        // As internal lints are currently always run if there are `unstable_options`,
        // they are added to the lint store of rustdoc. Internal lints are also
        // not used via the `lint_mod` query. Crate lints run outside of a query
        // so rustdoc currently doesn't disable them.
        //
        // Instead of relying on this, either change crate lints to a query disabled by
        // rustdoc, only run internal lints if the user is explicitly opting in
        // or figure out a different way to avoid running lints for rustdoc.
        if cx.tcx.sess.opts.actually_rustdoc {
            return;
        }

        let (span, def_id, substs) = match expr.kind {
            ExprKind::MethodCall(segment, _, _)
                if let Some(def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id) =>
            {
                (segment.ident.span, def_id, cx.typeck_results().node_substs(expr.hir_id))
            },
            _ => {
                let &ty::FnDef(def_id, substs) =
                    cx.typeck_results()
                        .node_type(expr.hir_id)
                        .kind() else { return };
                (expr.span, def_id, substs)
            }
        };
        if let Ok(Some(instance)) = ty::Instance::resolve(cx.tcx, cx.param_env, def_id, substs) {
            let def_id = instance.def_id();
            if cx.tcx.has_attr(def_id, sym::rustc_lint_query_instability) {
                cx.struct_span_lint(POTENTIAL_QUERY_INSTABILITY, span, |lint| {
                    let msg = format!(
                        "using `{}` can result in unstable query results",
                        cx.tcx.item_name(def_id)
                    );
                    lint.build(&msg)
                        .note("if you believe this case to be fine, allow this lint and add a comment explaining your rationale")
                        .emit();
                })
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
    fn check_path(&mut self, cx: &LateContext<'_>, path: &'tcx Path<'tcx>, _: HirId) {
        let segments = path.segments.iter().rev().skip(1).rev();

        if let Some(last) = segments.last() {
            let span = path.span.with_hi(last.ident.span.hi());
            if lint_ty_kind_usage(cx, last) {
                cx.struct_span_lint(USAGE_OF_TY_TYKIND, span, |lint| {
                    lint.build("usage of `ty::TyKind::<kind>`")
                        .span_suggestion(
                            span,
                            "try using ty::<kind> directly",
                            "ty".to_string(),
                            Applicability::MaybeIncorrect, // ty maybe needs an import
                        )
                        .emit();
                })
            }
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &'tcx Ty<'tcx>) {
        match &ty.kind {
            TyKind::Path(QPath::Resolved(_, path)) => {
                if let Some(last) = path.segments.iter().last() {
                    if lint_ty_kind_usage(cx, last) {
                        cx.struct_span_lint(USAGE_OF_TY_TYKIND, path.span, |lint| {
                            lint.build("usage of `ty::TyKind`")
                                .help("try using `Ty` instead")
                                .emit();
                        })
                    } else {
                        if ty.span.from_expansion() {
                            return;
                        }
                        if let Some(t) = is_ty_or_ty_ctxt(cx, ty) {
                            if path.segments.len() > 1 {
                                cx.struct_span_lint(USAGE_OF_QUALIFIED_TY, path.span, |lint| {
                                    lint.build(&format!("usage of qualified `ty::{}`", t))
                                        .span_suggestion(
                                            path.span,
                                            "try importing it and using it unqualified",
                                            t,
                                            // The import probably needs to be changed
                                            Applicability::MaybeIncorrect,
                                        )
                                        .emit();
                                })
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn lint_ty_kind_usage(cx: &LateContext<'_>, segment: &PathSegment<'_>) -> bool {
    if let Some(res) = segment.res {
        if let Some(did) = res.opt_def_id() {
            return cx.tcx.is_diagnostic_item(sym::TyKind, did);
        }
    }

    false
}

fn is_ty_or_ty_ctxt(cx: &LateContext<'_>, ty: &Ty<'_>) -> Option<String> {
    if let TyKind::Path(QPath::Resolved(_, path)) = &ty.kind {
        match path.res {
            Res::Def(_, def_id) => {
                if let Some(name @ (sym::Ty | sym::TyCtxt)) = cx.tcx.get_diagnostic_name(def_id) {
                    return Some(format!("{}{}", name, gen_args(path.segments.last().unwrap())));
                }
            }
            // Only lint on `&Ty` and `&TyCtxt` if it is used outside of a trait.
            Res::SelfTy { trait_: None, alias_to: Some((did, _)) } => {
                if let ty::Adt(adt, substs) = cx.tcx.type_of(did).kind() {
                    if let Some(name @ (sym::Ty | sym::TyCtxt)) =
                        cx.tcx.get_diagnostic_name(adt.did)
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
    }

    None
}

fn gen_args(segment: &PathSegment<'_>) -> String {
    if let Some(args) = &segment.args {
        let lifetimes = args
            .args
            .iter()
            .filter_map(|arg| {
                if let GenericArg::Lifetime(lt) = arg {
                    Some(lt.name.ident().to_string())
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
                        cx.struct_span_lint(
                            LINT_PASS_IMPL_WITHOUT_MACRO,
                            lint_pass.path.span,
                            |lint| {
                                lint.build("implementing `LintPass` by hand")
                                    .help("try using `declare_lint_pass!` or `impl_lint_pass!` instead")
                                    .emit();
                            },
                        )
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
                        let v = nested
                            .value_str()
                            .expect("#[doc(keyword = \"...\")] expected a value!");
                        if is_doc_keyword(v) {
                            return;
                        }
                        cx.struct_span_lint(EXISTING_DOC_KEYWORD, attr.span, |lint| {
                            lint.build(&format!(
                                "Found non-existing keyword `{}` used in \
                                     `#[doc(keyword = \"...\")]`",
                                v,
                            ))
                            .help("only existing keywords are allowed in core/std")
                            .emit();
                        });
                    }
                }
            }
        }
    }
}
