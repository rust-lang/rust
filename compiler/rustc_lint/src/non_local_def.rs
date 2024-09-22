use rustc_errors::MultiSpan;
use rustc_hir::def::DefKind;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Body, HirId, Item, ItemKind, Node, Path, QPath, TyKind};
use rustc_infer::infer::InferCtxt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_middle::ty::{
    self, Binder, EarlyBinder, TraitRef, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::symbol::kw;
use rustc_span::{ExpnKind, MacroKind, Span, Symbol, sym};
use rustc_trait_selection::error_reporting::traits::ambiguity::{
    CandidateSource, compute_applicable_impls_for_diagnostics,
};
use rustc_trait_selection::infer::TyCtxtInferExt;

use crate::lints::{NonLocalDefinitionsCargoUpdateNote, NonLocalDefinitionsDiag};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

declare_lint! {
    /// The `non_local_definitions` lint checks for `impl` blocks and `#[macro_export]`
    /// macro inside bodies (functions, enum discriminant, ...).
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(non_local_definitions)]
    /// trait MyTrait {}
    /// struct MyStruct;
    ///
    /// fn foo() {
    ///     impl MyTrait for MyStruct {}
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating non-local definitions go against expectation and can create discrepancies
    /// in tooling. It should be avoided. It may become deny-by-default in edition 2024
    /// and higher, see the tracking issue <https://github.com/rust-lang/rust/issues/120363>.
    ///
    /// An `impl` definition is non-local if it is nested inside an item and neither
    /// the type nor the trait are at the same nesting level as the `impl` block.
    ///
    /// All nested bodies (functions, enum discriminant, array length, consts) (expect for
    /// `const _: Ty = { ... }` in top-level module, which is still undecided) are checked.
    pub NON_LOCAL_DEFINITIONS,
    Allow,
    "checks for non-local definitions",
    report_in_external_macro
}

#[derive(Default)]
pub(crate) struct NonLocalDefinitions {
    body_depth: u32,
}

impl_lint_pass!(NonLocalDefinitions => [NON_LOCAL_DEFINITIONS]);

// FIXME(Urgau): Figure out how to handle modules nested in bodies.
// It's currently not handled by the current logic because modules are not bodies.
// They don't even follow the correct order (check_body -> check_mod -> check_body_post)
// instead check_mod is called after every body has been handled.

impl<'tcx> LateLintPass<'tcx> for NonLocalDefinitions {
    fn check_body(&mut self, _cx: &LateContext<'tcx>, _body: &Body<'tcx>) {
        self.body_depth += 1;
    }

    fn check_body_post(&mut self, _cx: &LateContext<'tcx>, _body: &Body<'tcx>) {
        self.body_depth -= 1;
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if self.body_depth == 0 {
            return;
        }

        let def_id = item.owner_id.def_id.into();
        let parent = cx.tcx.parent(def_id);
        let parent_def_kind = cx.tcx.def_kind(parent);
        let parent_opt_item_name = cx.tcx.opt_item_name(parent);

        // Per RFC we (currently) ignore anon-const (`const _: Ty = ...`) in top-level module.
        if self.body_depth == 1
            && parent_def_kind == DefKind::Const
            && parent_opt_item_name == Some(kw::Underscore)
        {
            return;
        }

        let cargo_update = || {
            let oexpn = item.span.ctxt().outer_expn_data();
            if let Some(def_id) = oexpn.macro_def_id
                && let ExpnKind::Macro(macro_kind, macro_name) = oexpn.kind
                && def_id.krate != LOCAL_CRATE
                && rustc_session::utils::was_invoked_from_cargo()
            {
                Some(NonLocalDefinitionsCargoUpdateNote {
                    macro_kind: macro_kind.descr(),
                    macro_name,
                    crate_name: cx.tcx.crate_name(def_id.krate),
                })
            } else {
                None
            }
        };

        // determining if we are in a doctest context can't currently be determined
        // by the code itself (there are no specific attributes), but fortunately rustdoc
        // sets a perma-unstable env var for libtest so we just reuse that for now
        let is_at_toplevel_doctest =
            || self.body_depth == 2 && std::env::var("UNSTABLE_RUSTDOC_TEST_PATH").is_ok();

        match item.kind {
            ItemKind::Impl(impl_) => {
                // The RFC states:
                //
                // > An item nested inside an expression-containing item (through any
                // > level of nesting) may not define an impl Trait for Type unless
                // > either the **Trait** or the **Type** is also nested inside the
                // > same expression-containing item.
                //
                // To achieve this we get try to get the paths of the _Trait_ and
                // _Type_, and we look inside those paths to try a find in one
                // of them a type whose parent is the same as the impl definition.
                //
                // If that's the case this means that this impl block declaration
                // is using local items and so we don't lint on it.

                // We also ignore anon-const in item by including the anon-const
                // parent as well.
                let parent_parent = if parent_def_kind == DefKind::Const
                    && parent_opt_item_name == Some(kw::Underscore)
                {
                    Some(cx.tcx.parent(parent))
                } else {
                    None
                };

                // Part 1: Is the Self type local?
                let self_ty_has_local_parent =
                    ty_has_local_parent(&impl_.self_ty.kind, cx, parent, parent_parent);

                if self_ty_has_local_parent {
                    return;
                }

                // Part 2: Is the Trait local?
                let of_trait_has_local_parent = impl_
                    .of_trait
                    .map(|of_trait| path_has_local_parent(of_trait.path, cx, parent, parent_parent))
                    .unwrap_or(false);

                if of_trait_has_local_parent {
                    return;
                }

                // Part 3: Is the impl definition leaking outside it's defining scope?
                //
                // We always consider inherent impls to be leaking.
                let impl_has_enough_non_local_candidates = cx
                    .tcx
                    .impl_trait_ref(def_id)
                    .map(|binder| {
                        impl_trait_ref_has_enough_non_local_candidates(
                            cx.tcx,
                            item.span,
                            def_id,
                            binder,
                            |did| did_has_local_parent(did, cx.tcx, parent, parent_parent),
                        )
                    })
                    .unwrap_or(false);

                if impl_has_enough_non_local_candidates {
                    return;
                }

                // Get the span of the parent const item ident (if it's a not a const anon).
                //
                // Used to suggest changing the const item to a const anon.
                let span_for_const_anon_suggestion = if parent_def_kind == DefKind::Const
                    && parent_opt_item_name != Some(kw::Underscore)
                    && let Some(parent) = parent.as_local()
                    && let Node::Item(item) = cx.tcx.hir_node_by_def_id(parent)
                    && let ItemKind::Const(ty, _, _) = item.kind
                    && let TyKind::Tup(&[]) = ty.kind
                {
                    Some(item.ident.span)
                } else {
                    None
                };

                let const_anon = matches!(parent_def_kind, DefKind::Const | DefKind::Static { .. })
                    .then_some(span_for_const_anon_suggestion);

                let may_remove = match &impl_.self_ty.kind {
                    TyKind::Ptr(mut_ty) | TyKind::Ref(_, mut_ty)
                        if ty_has_local_parent(&mut_ty.ty.kind, cx, parent, parent_parent) =>
                    {
                        let type_ =
                            if matches!(impl_.self_ty.kind, TyKind::Ptr(_)) { "*" } else { "&" };
                        let part = format!("{}{}", type_, mut_ty.mutbl.prefix_str());
                        Some((impl_.self_ty.span.shrink_to_lo().until(mut_ty.ty.span), part))
                    }
                    _ => None,
                };

                let impl_span = item.span.shrink_to_lo().to(impl_.self_ty.span);
                let mut ms = MultiSpan::from_span(impl_span);

                let (self_ty_span, self_ty_str) =
                    self_ty_kind_for_diagnostic(&impl_.self_ty, cx.tcx);

                ms.push_span_label(
                    self_ty_span,
                    fluent::lint_non_local_definitions_self_ty_not_local,
                );
                let of_trait_str = if let Some(of_trait) = &impl_.of_trait {
                    ms.push_span_label(
                        path_span_without_args(&of_trait.path),
                        fluent::lint_non_local_definitions_of_trait_not_local,
                    );
                    Some(path_name_to_string(&of_trait.path))
                } else {
                    None
                };

                let (doctest, move_to) = if is_at_toplevel_doctest() {
                    (true, None)
                } else {
                    let mut collector = PathCollector { paths: Vec::new() };
                    collector.visit_ty(&impl_.self_ty);
                    if let Some(of_trait) = &impl_.of_trait {
                        collector.visit_trait_ref(of_trait);
                    }
                    collector.visit_generics(&impl_.generics);

                    let mut may_move: Vec<Span> = collector
                        .paths
                        .into_iter()
                        .filter_map(|path| {
                            if let Some(did) = path.res.opt_def_id()
                                && did_has_local_parent(did, cx.tcx, parent, parent_parent)
                            {
                                Some(cx.tcx.def_span(did))
                            } else {
                                None
                            }
                        })
                        .collect();
                    may_move.sort();
                    may_move.dedup();

                    let move_to = if may_move.is_empty() {
                        ms.push_span_label(
                            cx.tcx.def_span(parent),
                            fluent::lint_non_local_definitions_impl_move_help,
                        );
                        None
                    } else {
                        Some((cx.tcx.def_span(parent), may_move))
                    };

                    (false, move_to)
                };

                let macro_to_change =
                    if let ExpnKind::Macro(kind, name) = item.span.ctxt().outer_expn_data().kind {
                        Some((name.to_string(), kind.descr()))
                    } else {
                        None
                    };

                cx.emit_span_lint(NON_LOCAL_DEFINITIONS, ms, NonLocalDefinitionsDiag::Impl {
                    depth: self.body_depth,
                    body_kind_descr: cx.tcx.def_kind_descr(parent_def_kind, parent),
                    body_name: parent_opt_item_name
                        .map(|s| s.to_ident_string())
                        .unwrap_or_else(|| "<unnameable>".to_string()),
                    cargo_update: cargo_update(),
                    const_anon,
                    self_ty_str,
                    of_trait_str,
                    move_to,
                    doctest,
                    may_remove,
                    has_trait: impl_.of_trait.is_some(),
                    macro_to_change,
                })
            }
            ItemKind::Macro(_macro, MacroKind::Bang)
                if cx.tcx.has_attr(item.owner_id.def_id, sym::macro_export) =>
            {
                cx.emit_span_lint(
                    NON_LOCAL_DEFINITIONS,
                    item.span,
                    NonLocalDefinitionsDiag::MacroRules {
                        depth: self.body_depth,
                        body_kind_descr: cx.tcx.def_kind_descr(parent_def_kind, parent),
                        body_name: parent_opt_item_name
                            .map(|s| s.to_ident_string())
                            .unwrap_or_else(|| "<unnameable>".to_string()),
                        cargo_update: cargo_update(),
                        doctest: is_at_toplevel_doctest(),
                    },
                )
            }
            _ => {}
        }
    }
}

// Detecting if the impl definition is leaking outside of its defining scope.
//
// Rule: for each impl, instantiate all local types with inference vars and
// then assemble candidates for that goal, if there are more than 1 (non-private
// impls), it does not leak.
//
// https://github.com/rust-lang/rust/issues/121621#issuecomment-1976826895
fn impl_trait_ref_has_enough_non_local_candidates<'tcx>(
    tcx: TyCtxt<'tcx>,
    infer_span: Span,
    trait_def_id: DefId,
    binder: EarlyBinder<'tcx, TraitRef<'tcx>>,
    mut did_has_local_parent: impl FnMut(DefId) -> bool,
) -> bool {
    let infcx = tcx
        .infer_ctxt()
        // We use the new trait solver since the obligation we are trying to
        // prove here may overflow and those are fatal in the old trait solver.
        // Which is unacceptable for a lint.
        //
        // Thanksfully the part we use here are very similar to the
        // new-trait-solver-as-coherence, which is in stabilization.
        //
        // https://github.com/rust-lang/rust/issues/123573
        .with_next_trait_solver(true)
        .build();

    let trait_ref = binder.instantiate(tcx, infcx.fresh_args_for_item(infer_span, trait_def_id));

    let trait_ref = trait_ref.fold_with(&mut ReplaceLocalTypesWithInfer {
        infcx: &infcx,
        infer_span,
        did_has_local_parent: &mut did_has_local_parent,
    });

    let poly_trait_obligation = Obligation::new(
        tcx,
        ObligationCause::dummy(),
        ty::ParamEnv::empty(),
        Binder::dummy(trait_ref),
    );

    let ambiguities = compute_applicable_impls_for_diagnostics(&infcx, &poly_trait_obligation);

    let mut it = ambiguities.iter().filter(|ambi| match ambi {
        CandidateSource::DefId(did) => !did_has_local_parent(*did),
        CandidateSource::ParamEnv(_) => unreachable!(),
    });

    let _ = it.next();
    it.next().is_some()
}

/// Replace every local type by inference variable.
///
/// ```text
/// <Global<Local> as std::cmp::PartialEq<Global<Local>>>
/// to
/// <Global<_> as std::cmp::PartialEq<Global<_>>>
/// ```
struct ReplaceLocalTypesWithInfer<'a, 'tcx, F: FnMut(DefId) -> bool> {
    infcx: &'a InferCtxt<'tcx>,
    did_has_local_parent: F,
    infer_span: Span,
}

impl<'a, 'tcx, F: FnMut(DefId) -> bool> TypeFolder<TyCtxt<'tcx>>
    for ReplaceLocalTypesWithInfer<'a, 'tcx, F>
{
    fn cx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        if let Some(def) = t.ty_adt_def()
            && (self.did_has_local_parent)(def.did())
        {
            self.infcx.next_ty_var(self.infer_span)
        } else {
            t.super_fold_with(self)
        }
    }
}

/// Simple hir::Path collector
struct PathCollector<'tcx> {
    paths: Vec<Path<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for PathCollector<'tcx> {
    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        self.paths.push(path.clone()); // need to clone, bc of the restricted lifetime
        intravisit::walk_path(self, path)
    }
}

/// Given a `Ty` we check if the (outermost) type is local.
fn ty_has_local_parent(
    ty_kind: &TyKind<'_>,
    cx: &LateContext<'_>,
    impl_parent: DefId,
    impl_parent_parent: Option<DefId>,
) -> bool {
    match ty_kind {
        TyKind::Path(QPath::Resolved(_, ty_path)) => {
            path_has_local_parent(ty_path, cx, impl_parent, impl_parent_parent)
        }
        TyKind::TraitObject([principle_poly_trait_ref, ..], _, _) => path_has_local_parent(
            principle_poly_trait_ref.0.trait_ref.path,
            cx,
            impl_parent,
            impl_parent_parent,
        ),
        TyKind::TraitObject([], _, _)
        | TyKind::InferDelegation(_, _)
        | TyKind::Slice(_)
        | TyKind::Array(_, _)
        | TyKind::Ptr(_)
        | TyKind::Ref(_, _)
        | TyKind::BareFn(_)
        | TyKind::Never
        | TyKind::Tup(_)
        | TyKind::Path(_)
        | TyKind::Pat(..)
        | TyKind::AnonAdt(_)
        | TyKind::OpaqueDef(_, _, _)
        | TyKind::Typeof(_)
        | TyKind::Infer
        | TyKind::Err(_) => false,
    }
}

/// Given a path and a parent impl def id, this checks if the if parent resolution
/// def id correspond to the def id of the parent impl definition.
///
/// Given this path, we will look at the path (and ignore any generic args):
///
/// ```text
///    std::convert::PartialEq<Foo<Bar>>
///    ^^^^^^^^^^^^^^^^^^^^^^^
/// ```
#[inline]
fn path_has_local_parent(
    path: &Path<'_>,
    cx: &LateContext<'_>,
    impl_parent: DefId,
    impl_parent_parent: Option<DefId>,
) -> bool {
    path.res
        .opt_def_id()
        .is_some_and(|did| did_has_local_parent(did, cx.tcx, impl_parent, impl_parent_parent))
}

/// Given a def id and a parent impl def id, this checks if the parent
/// def id (modulo modules) correspond to the def id of the parent impl definition.
#[inline]
fn did_has_local_parent(
    did: DefId,
    tcx: TyCtxt<'_>,
    impl_parent: DefId,
    impl_parent_parent: Option<DefId>,
) -> bool {
    did.is_local()
        && if let Some(did_parent) = tcx.opt_parent(did) {
            did_parent == impl_parent
                || Some(did_parent) == impl_parent_parent
                || !did_parent.is_crate_root()
                    && tcx.def_kind(did_parent) == DefKind::Mod
                    && did_has_local_parent(did_parent, tcx, impl_parent, impl_parent_parent)
        } else {
            false
        }
}

/// Return for a given `Path` the span until the last args
fn path_span_without_args(path: &Path<'_>) -> Span {
    if let Some(args) = &path.segments.last().unwrap().args {
        path.span.until(args.span_ext)
    } else {
        path.span
    }
}

/// Return a "error message-able" ident for the last segment of the `Path`
fn path_name_to_string(path: &Path<'_>) -> String {
    path.segments.last().unwrap().ident.name.to_ident_string()
}

/// Compute the `Span` and visual representation for the `Self` we want to point at;
/// It follows part of the actual logic of non-local, and if possible return the least
/// amount possible for the span and representation.
fn self_ty_kind_for_diagnostic(ty: &rustc_hir::Ty<'_>, tcx: TyCtxt<'_>) -> (Span, String) {
    match ty.kind {
        TyKind::Path(QPath::Resolved(_, ty_path)) => (
            path_span_without_args(ty_path),
            ty_path
                .res
                .opt_def_id()
                .map(|did| tcx.opt_item_name(did))
                .flatten()
                .as_ref()
                .map(|s| Symbol::as_str(s))
                .unwrap_or("<unnameable>")
                .to_string(),
        ),
        TyKind::TraitObject([principle_poly_trait_ref, ..], _, _) => {
            let path = &principle_poly_trait_ref.0.trait_ref.path;
            (
                path_span_without_args(path),
                path.res
                    .opt_def_id()
                    .map(|did| tcx.opt_item_name(did))
                    .flatten()
                    .as_ref()
                    .map(|s| Symbol::as_str(s))
                    .unwrap_or("<unnameable>")
                    .to_string(),
            )
        }
        _ => (ty.span, rustc_hir_pretty::ty_to_string(&tcx, ty)),
    }
}
