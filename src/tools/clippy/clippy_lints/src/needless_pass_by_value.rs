use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ptr::get_spans;
use clippy_utils::source::{SpanRangeExt, snippet};
use clippy_utils::ty::{
    implements_trait, implements_trait_with_env_from_iter, is_copy, is_type_diagnostic_item, is_type_lang_item,
};
use clippy_utils::{is_self, peel_hir_ty_options};
use rustc_abi::ExternAbi;
use rustc_errors::{Applicability, Diag};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Attribute, BindingMode, Body, FnDecl, GenericArg, HirId, HirIdSet, Impl, ItemKind, LangItem, Mutability, Node,
    PatKind, QPath, TyKind,
};
use rustc_hir_typeck::expr_use_visitor as euv;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::FakeReadCause;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::kw;
use rustc_span::{Span, sym};
use rustc_trait_selection::traits;
use rustc_trait_selection::traits::misc::type_allowed_to_implement_copy;

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
    /// ```no_run
    /// fn foo(v: Vec<i32>) {
    ///     assert_eq!(v.len(), 42);
    /// }
    /// ```
    /// should be
    /// ```no_run
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

        let hir_id = cx.tcx.local_def_id_to_hir_id(fn_def_id);

        match kind {
            FnKind::ItemFn(.., header) => {
                let attrs = cx.tcx.hir_attrs(hir_id);
                if header.abi != ExternAbi::Rust || requires_exact_signature(attrs) {
                    return;
                }
            },
            FnKind::Method(..) => (),
            FnKind::Closure => return,
        }

        // Exclude non-inherent impls
        if let Node::Item(item) = cx.tcx.parent_hir_node(hir_id)
            && matches!(
                item.kind,
                ItemKind::Impl(Impl { of_trait: Some(_), .. }) | ItemKind::Trait(..)
            )
        {
            return;
        }

        // Allow `Borrow` or functions to be taken by value
        let allowed_traits = [
            need!(cx.tcx.lang_items().fn_trait()),
            need!(cx.tcx.lang_items().fn_once_trait()),
            need!(cx.tcx.lang_items().fn_mut_trait()),
            need!(cx.tcx.get_diagnostic_item(sym::RangeBounds)),
        ];

        let sized_trait = need!(cx.tcx.lang_items().sized_trait());
        let meta_sized_trait = need!(cx.tcx.lang_items().meta_sized_trait());

        let preds = traits::elaborate(cx.tcx, cx.param_env.caller_bounds().iter())
            .filter(|p| !p.is_global())
            .filter_map(|pred| {
                // Note that we do not want to deal with qualified predicates here.
                match pred.kind().no_bound_vars() {
                    Some(ty::ClauseKind::Trait(pred))
                        if pred.def_id() != sized_trait && pred.def_id() != meta_sized_trait =>
                    {
                        Some(pred)
                    },
                    _ => None,
                }
            })
            .collect::<Vec<_>>();

        // Collect moved variables and spans which will need dereferencing from the
        // function body.
        let MovedVariablesCtxt { moved_vars } = {
            let mut ctx = MovedVariablesCtxt::default();
            euv::ExprUseVisitor::for_clippy(cx, fn_def_id, &mut ctx)
                .consume_body(body)
                .into_ok();
            ctx
        };

        let fn_sig = cx.tcx.fn_sig(fn_def_id).instantiate_identity();
        let fn_sig = cx.tcx.liberate_late_bound_regions(fn_def_id.to_def_id(), fn_sig);

        for (idx, ((input, &ty), arg)) in decl.inputs.iter().zip(fn_sig.inputs()).zip(body.params).enumerate() {
            // All spans generated from a proc-macro invocation are the same...
            if span == input.span {
                return;
            }

            // Ignore `self`s and params whose variable name starts with an underscore
            if let PatKind::Binding(.., ident, _) = arg.pat.kind {
                if idx == 0 && ident.name == kw::SelfLower {
                    continue;
                }
                if ident.name.as_str().starts_with('_') {
                    continue;
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
                        let ty_empty_region = Ty::new_imm_ref(cx.tcx, cx.tcx.lifetimes.re_erased, ty);
                        preds.iter().all(|t| {
                            let ty_params = t.trait_ref.args.iter().skip(1).collect::<Vec<_>>();
                            implements_trait(cx, ty_empty_region, t.def_id(), &ty_params)
                        })
                    },
                )
            };

            if !is_self(arg)
                && !ty.is_mutable_ptr()
                && !is_copy(cx, ty)
                && ty.is_sized(cx.tcx, cx.typing_env())
                && !allowed_traits.iter().any(|&t| {
                    implements_trait_with_env_from_iter(
                        cx.tcx,
                        cx.typing_env(),
                        ty,
                        t,
                        None,
                        [None::<ty::GenericArg<'tcx>>],
                    )
                })
                && !implements_borrow_trait
                && !all_borrowable_trait
                && let PatKind::Binding(BindingMode(_, Mutability::Not), canonical_id, ..) = arg.pat.kind
                && !moved_vars.contains(&canonical_id)
            {
                // Dereference suggestion
                let sugg = |diag: &mut Diag<'_, ()>| {
                    if let ty::Adt(def, ..) = ty.kind()
                        && let Some(span) = cx.tcx.hir_span_if_local(def.did())
                        && type_allowed_to_implement_copy(
                            cx.tcx,
                            cx.param_env,
                            ty,
                            traits::ObligationCause::dummy_with_span(span),
                            rustc_hir::Safety::Safe,
                        )
                        .is_ok()
                    {
                        diag.span_help(span, "or consider marking this type as `Copy`");
                    }

                    if is_type_diagnostic_item(cx, ty, sym::Vec)
                        && let Some(clone_spans) = get_spans(cx, Some(body.id()), idx, &[(sym::clone, ".to_owned()")])
                        && let TyKind::Path(QPath::Resolved(_, path)) = input.kind
                        && let Some(elem_ty) = path
                            .segments
                            .iter()
                            .find(|seg| seg.ident.name == sym::Vec)
                            .and_then(|ps| ps.args.as_ref())
                            .map(|params| {
                                params
                                    .args
                                    .iter()
                                    .find_map(|arg| match arg {
                                        GenericArg::Type(ty) => Some(ty),
                                        _ => None,
                                    })
                                    .unwrap()
                            })
                    {
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
                                span.get_source_text(cx).map_or_else(
                                    || "change the call to".to_owned(),
                                    |src| format!("change `{src}` to"),
                                ),
                                suggestion,
                                Applicability::Unspecified,
                            );
                        }

                        // cannot be destructured, no need for `*` suggestion
                        return;
                    }

                    if is_type_lang_item(cx, ty, LangItem::String)
                        && let Some(clone_spans) = get_spans(
                            cx,
                            Some(body.id()),
                            idx,
                            &[(sym::clone, ".to_string()"), (sym::as_str, "")],
                        )
                    {
                        diag.span_suggestion(
                            input.span,
                            "consider changing the type to",
                            "&str",
                            Applicability::Unspecified,
                        );

                        for (span, suggestion) in clone_spans {
                            diag.span_suggestion(
                                span,
                                span.get_source_text(cx).map_or_else(
                                    || "change the call to".to_owned(),
                                    |src| format!("change `{src}` to"),
                                ),
                                suggestion,
                                Applicability::Unspecified,
                            );
                        }

                        return;
                    }

                    diag.span_suggestion_verbose(
                        peel_hir_ty_options(cx, input).span.shrink_to_lo(),
                        "consider taking a reference instead",
                        '&',
                        Applicability::MaybeIncorrect,
                    );
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

/// Functions marked with these attributes must have the exact signature.
pub(crate) fn requires_exact_signature(attrs: &[Attribute]) -> bool {
    attrs.iter().any(Attribute::is_proc_macro_attr)
}

#[derive(Default)]
struct MovedVariablesCtxt {
    moved_vars: HirIdSet,
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

    fn use_cloned(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId) {}

    fn borrow(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId, _: ty::BorrowKind) {}

    fn mutate(&mut self, _: &euv::PlaceWithHirId<'tcx>, _: HirId) {}

    fn fake_read(&mut self, _: &rustc_hir_typeck::expr_use_visitor::PlaceWithHirId<'tcx>, _: FakeReadCause, _: HirId) {}
}
