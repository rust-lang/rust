use clippy_utils::diagnostics::{multispan_sugg, span_lint_and_then};
use clippy_utils::ptr::get_spans;
use clippy_utils::source::{snippet, snippet_opt};
use clippy_utils::ty::{
    implements_trait, implements_trait_with_env, is_copy, is_type_diagnostic_item, is_type_lang_item,
};
use clippy_utils::{get_trait_def_id, is_self, paths};
use if_chain::if_chain;
use rustc_ast::ast::Attribute;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, Diagnostic};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    BindingAnnotation, Body, FnDecl, GenericArg, HirId, Impl, ItemKind, Mutability, Node, PatKind, QPath, TyKind,
};
use rustc_hir::{HirIdMap, HirIdSet, LangItem};
use rustc_hir_typeck::expr_use_visitor as euv;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, TypeVisitableExt};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::kw;
use rustc_span::{sym, Span};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::misc::type_allowed_to_implement_copy;
use std::borrow::Cow;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions taking arguments by value, but not
    /// consuming them in its
    /// body.
    ///
    /// ### Why is this bad?
    /// Taking arguments by reference is more flexible and can
    /// sometimes avoid
    /// unnecessary allocations.
    ///
    /// ### Known problems
    /// * This lint suggests taking an argument by reference,
    /// however sometimes it is better to let users decide the argument type
    /// (by using `Borrow` trait, for example), depending on how the function is used.
    ///
    /// ### Example
    /// ```rust
    /// fn foo(v: Vec<i32>) {
    ///     assert_eq!(v.len(), 42);
    /// }
    /// ```
    /// should be
    /// ```rust
    /// fn foo(v: &[i32]) {
    ///     assert_eq!(v.len(), 42);
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEEDLESS_PASS_BY_VALUE,
    pedantic,
    "functions taking arguments by value, but not consuming them in its body"
}

declare_lint_pass!(NeedlessPassByValue => [NEEDLESS_PASS_BY_VALUE]);

macro_rules! need {
    ($e: expr) => {
        if let Some(x) = $e {
            x
        } else {
            return;
        }
    };
}

impl<'tcx> LateLintPass<'tcx> for NeedlessPassByValue {
    #[expect(clippy::too_many_lines)]
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl<'_>,
        body: &'tcx Body<'_>,
        span: Span,
        fn_def_id: LocalDefId,
    ) {
        if span.from_expansion() {
            return;
        }

        let hir_id = cx.tcx.hir().local_def_id_to_hir_id(fn_def_id);

        match kind {
            FnKind::ItemFn(.., header) => {
                let attrs = cx.tcx.hir().attrs(hir_id);
                if header.abi != Abi::Rust || requires_exact_signature(attrs) {
                    return;
                }
            },
            FnKind::Method(..) => (),
            FnKind::Closure => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir().find_parent(hir_id) {
            if matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            ) {
                return;
            }
        }

        // Allow `Borrow` or functions to be taken by value
        let allowed_traits = [
            need!(cx.tcx.lang_items().fn_trait()),
            need!(cx.tcx.lang_items().fn_once_trait()),
            need!(cx.tcx.lang_items().fn_mut_trait()),
            need!(get_trait_def_id(cx, &paths::RANGE_ARGUMENT_TRAIT)),
        ];

        let sized_trait = need!(cx.tcx.lang_items().sized_trait());

        let preds = traits::elaborate_predicates(cx.tcx, cx.param_env.caller_bounds().iter())
            .filter(|p| !p.is_global())
            .filter_map(|obligation| {
                // Note that we do not want to deal with qualified predicates here.
                match obligation.predicate.kind().no_bound_vars() {
                    Some(ty::PredicateKind::Clause(ty::Clause::Trait(pred))) if pred.def_id() != sized_trait => {
                        Some(pred)
                    },
                    _ => None,
                }
            })
            .collect::<Vec<_>>();

        // Collect moved variables and spans which will need dereferencings from the
        // function body.
        let MovedVariablesCtxt {
            moved_vars,
            spans_need_deref,
            ..
        } = {
            let mut ctx = MovedVariablesCtxt::default();
            let infcx = cx.tcx.infer_ctxt().build();
            euv::ExprUseVisitor::new(&mut ctx, &infcx, fn_def_id, cx.param_env, cx.typeck_results()).consume_body(body);
            ctx
        };

        let fn_sig = cx.tcx.fn_sig(fn_def_id).subst_identity();
        let fn_sig = cx.tcx.liberate_late_bound_regions(fn_def_id.to_def_id(), fn_sig);

        for (idx, ((input, &ty), arg)) in decl.inputs.iter().zip(fn_sig.inputs()).zip(body.params).enumerate() {
            // All spans generated from a proc-macro invocation are the same...
            if span == input.span {
                return;
            }

            // Ignore `self`s.
            if idx == 0 {
                if let PatKind::Binding(.., ident, _) = arg.pat.kind {
                    if ident.name == kw::SelfLower {
                        continue;
                    }
                }
            }

            //
            // * Exclude a type that is specifically bounded by `Borrow`.
            // * Exclude a type whose reference also fulfills its bound. (e.g., `std::convert::AsRef`,
            //   `serde::Serialize`)
            let (implements_borrow_trait, all_borrowable_trait) = {
                let preds = preds.iter().filter(|t| t.self_ty() == ty).collect::<Vec<_>>();

                (
                    preds.iter().any(|t| cx.tcx.is_diagnostic_item(sym::Borrow, t.def_id())),
                    !preds.is_empty() && {
                        let ty_empty_region = cx.tcx.mk_imm_ref(cx.tcx.lifetimes.re_erased, ty);
                        preds.iter().all(|t| {
                            let ty_params = t.trait_ref.substs.iter().skip(1).collect::<Vec<_>>();
                            implements_trait(cx, ty_empty_region, t.def_id(), &ty_params)
                        })
                    },
                )
            };

            if_chain! {
                if !is_self(arg);
                if !ty.is_mutable_ptr();
                if !is_copy(cx, ty);
                if ty.is_sized(cx.tcx, cx.param_env);
                if !allowed_traits.iter().any(|&t| implements_trait_with_env(cx.tcx, cx.param_env, ty, t, [None]));
                if !implements_borrow_trait;
                if !all_borrowable_trait;

                if let PatKind::Binding(BindingAnnotation(_, Mutability::Not), canonical_id, ..) = arg.pat.kind;
                if !moved_vars.contains(&canonical_id);
                then {
                    // Dereference suggestion
                    let sugg = |diag: &mut Diagnostic| {
                        if let ty::Adt(def, ..) = ty.kind() {
                            if let Some(span) = cx.tcx.hir().span_if_local(def.did()) {
                                if type_allowed_to_implement_copy(
                                    cx.tcx,
                                    cx.param_env,
                                    ty,
                                    traits::ObligationCause::dummy_with_span(span),
                                ).is_ok() {
                                    diag.span_help(span, "consider marking this type as `Copy`");
                                }
                            }
                        }

                        let deref_span = spans_need_deref.get(&canonical_id);
                        if_chain! {
                            if is_type_diagnostic_item(cx, ty, sym::Vec);
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_owned()")]);
                            if let TyKind::Path(QPath::Resolved(_, path)) = input.kind;
                            if let Some(elem_ty) = path.segments.iter()
                                .find(|seg| seg.ident.name == sym::Vec)
                                .and_then(|ps| ps.args.as_ref())
                                .map(|params| params.args.iter().find_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                }).unwrap());
                            then {
                                let slice_ty = format!("&[{}]", snippet(cx, elem_ty.span, "_"));
                                diag.span_suggestion(
                                    input.span,
                                    "consider changing the type to",
                                    slice_ty,
                                    Applicability::Unspecified,
                                );

                                for (span, suggestion) in clone_spans {
                                    diag.span_suggestion(
                                        span,
                                        snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{x}` to")),
                                            )
                                            .as_ref(),
                                        suggestion,
                                        Applicability::Unspecified,
                                    );
                                }

                                // cannot be destructured, no need for `*` suggestion
                                assert!(deref_span.is_none());
                                return;
                            }
                        }

                        if is_type_lang_item(cx, ty, LangItem::String) {
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_string()"), ("as_str", "")]) {
                                diag.span_suggestion(
                                    input.span,
                                    "consider changing the type to",
                                    "&str",
                                    Applicability::Unspecified,
                                );

                                for (span, suggestion) in clone_spans {
                                    diag.span_suggestion(
                                        span,
                                        snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{x}` to"))
                                            )
                                            .as_ref(),
                                        suggestion,
                                        Applicability::Unspecified,
                                    );
                                }

                                assert!(deref_span.is_none());
                                return;
                            }
                        }

                        let mut spans = vec![(input.span, format!("&{}", snippet(cx, input.span, "_")))];

                        // Suggests adding `*` to dereference the added reference.
                        if let Some(deref_span) = deref_span {
                            spans.extend(
                                deref_span
                                    .iter()
                                    .copied()
                                    .map(|span| (span, format!("*{}", snippet(cx, span, "<expr>")))),
                            );
                            spans.sort_by_key(|&(span, _)| span);
                        }
                        multispan_sugg(diag, "consider taking a reference instead", spans);
                    };

                    span_lint_and_then(
                        cx,
                        NEEDLESS_PASS_BY_VALUE,
                        input.span,
                        "this argument is passed by value, but not consumed in the function body",
                        sugg,
                    );
                }
            }
        }
    }
}

/// Functions marked with these attributes must have the exact signature.
fn requires_exact_signature(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
            .iter()
            .any(|&allow| attr.has_name(allow))
    })
}

#[derive(Default)]
struct MovedVariablesCtxt {
    moved_vars: HirIdSet,
    /// Spans which need to be prefixed with `*` for dereferencing the
    /// suggested additional reference.
    spans_need_deref: HirIdMap<FxHashSet<Span>>,
}

impl MovedVariablesCtxt {
    fn move_common(&mut self, cmt: &euv::PlaceWithHirId<'_>) {
        if let euv::PlaceBase::Local(vid) = cmt.place.base {
            self.moved_vars.insert(vid);
        }
    }
}

impl<'tcx> euv::Delegate<'tcx> for MovedVariablesCtxt {
    fn consume(&mut self, cmt: &euv::PlaceWithHirId<'tcx>, _: HirId) {
        self.move_common(cmt);
    }

    fn borrow(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {}

    fn mutate(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &rustc_hir_typeck::expr_use_visitor::PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}
