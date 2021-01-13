//! Some lints that are only useful in the compiler or crates that use compiler internals, such as
//! Clippy.

use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast::{Item, ItemKind};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def::Res;
use rustc_hir::{GenericArg, HirId, MutTy, Mutability, Path, PathSegment, QPath, Ty, TyKind};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint, impl_lint_pass};
use rustc_span::hygiene::{ExpnKind, MacroKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};

declare_tool_lint! {
    pub rustc::DEFAULT_HASH_TYPES,
    Allow,
    "forbid HashMap and HashSet and suggest the FxHash* variants",
    report_in_external_macro: true
}

pub struct DefaultHashTypes {
    map: FxHashMap<Symbol, Symbol>,
}

impl DefaultHashTypes {
    // we are allowed to use `HashMap` and `HashSet` as identifiers for implementing the lint itself
    #[allow(rustc::default_hash_types)]
    pub fn new() -> Self {
        let mut map = FxHashMap::default();
        map.insert(sym::HashMap, sym::FxHashMap);
        map.insert(sym::HashSet, sym::FxHashSet);
        Self { map }
    }
}

impl_lint_pass!(DefaultHashTypes => [DEFAULT_HASH_TYPES]);

impl EarlyLintPass for DefaultHashTypes {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: Ident) {
        if let Some(replace) = self.map.get(&ident.name) {
            cx.struct_span_lint(DEFAULT_HASH_TYPES, ident.span, |lint| {
                // FIXME: We can avoid a copy here. Would require us to take String instead of &str.
                let msg = format!("Prefer {} over {}, it has better performance", replace, ident);
                lint.build(&msg)
                    .span_suggestion(
                        ident.span,
                        "use",
                        replace.to_string(),
                        Applicability::MaybeIncorrect, // FxHashMap, ... needs another import
                    )
                    .note(&format!(
                        "a `use rustc_data_structures::fx::{}` may be necessary",
                        replace
                    ))
                    .emit();
            });
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
    pub rustc::TY_PASS_BY_REFERENCE,
    Allow,
    "passing `Ty` or `TyCtxt` by reference",
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
    TY_PASS_BY_REFERENCE,
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
            TyKind::Path(qpath) => {
                if let QPath::Resolved(_, path) = qpath {
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
                                                "try using it unqualified",
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
            }
            TyKind::Rptr(_, MutTy { ty: inner_ty, mutbl: Mutability::Not }) => {
                if let Some(impl_did) = cx.tcx.impl_of_method(ty.hir_id.owner.to_def_id()) {
                    if cx.tcx.impl_trait_ref(impl_did).is_some() {
                        return;
                    }
                }
                if let Some(t) = is_ty_or_ty_ctxt(cx, &inner_ty) {
                    cx.struct_span_lint(TY_PASS_BY_REFERENCE, ty.span, |lint| {
                        lint.build(&format!("passing `{}` by reference", t))
                            .span_suggestion(
                                ty.span,
                                "try passing by value",
                                t,
                                // Changing type of function argument
                                Applicability::MaybeIncorrect,
                            )
                            .emit();
                    })
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
    if let TyKind::Path(qpath) = &ty.kind {
        if let QPath::Resolved(_, path) = qpath {
            match path.res {
                Res::Def(_, did) => {
                    if cx.tcx.is_diagnostic_item(sym::Ty, did) {
                        return Some(format!("Ty{}", gen_args(path.segments.last().unwrap())));
                    } else if cx.tcx.is_diagnostic_item(sym::TyCtxt, did) {
                        return Some(format!("TyCtxt{}", gen_args(path.segments.last().unwrap())));
                    }
                }
                // Only lint on `&Ty` and `&TyCtxt` if it is used outside of a trait.
                Res::SelfTy(None, Some((did, _))) => {
                    if let ty::Adt(adt, substs) = cx.tcx.type_of(did).kind() {
                        if cx.tcx.is_diagnostic_item(sym::Ty, adt.did) {
                            // NOTE: This path is currently unreachable as `Ty<'tcx>` is
                            // defined as a type alias meaning that `impl<'tcx> Ty<'tcx>`
                            // is not actually allowed.
                            //
                            // I(@lcnr) still kept this branch in so we don't miss this
                            // if we ever change it in the future.
                            return Some(format!("Ty<{}>", substs[0]));
                        } else if cx.tcx.is_diagnostic_item(sym::TyCtxt, adt.did) {
                            return Some(format!("TyCtxt<{}>", substs[0]));
                        }
                    }
                }
                _ => (),
            }
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
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Impl { of_trait: Some(lint_pass), .. } = &item.kind {
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
        for attr in item.attrs {
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
