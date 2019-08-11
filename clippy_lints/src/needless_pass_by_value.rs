use crate::utils::ptr::get_spans;
use crate::utils::{
    get_trait_def_id, implements_trait, in_macro_or_desugar, is_copy, is_self, match_type, multispan_sugg, paths,
    snippet, snippet_opt, span_lint_and_then,
};
use if_chain::if_chain;
use matches::matches;
use rustc::hir::intravisit::FnKind;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::middle::expr_use_visitor as euv;
use rustc::middle::mem_categorization as mc;
use rustc::traits;
use rustc::ty::{self, RegionKind, TypeFoldable};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_target::spec::abi::Abi;
use std::borrow::Cow;
use syntax::ast::Attribute;
use syntax::errors::DiagnosticBuilder;
use syntax_pos::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for functions taking arguments by value, but not
    /// consuming them in its
    /// body.
    ///
    /// **Why is this bad?** Taking arguments by reference is more flexible and can
    /// sometimes avoid
    /// unnecessary allocations.
    ///
    /// **Known problems:**
    /// * This lint suggests taking an argument by reference,
    /// however sometimes it is better to let users decide the argument type
    /// (by using `Borrow` trait, for example), depending on how the function is used.
    ///
    /// **Example:**
    /// ```rust
    /// fn foo(v: Vec<i32>) {
    ///     assert_eq!(v.len(), 42);
    /// }
    /// ```
    ///
    /// ```rust
    /// // should be
    /// fn foo(v: &[i32]) {
    ///     assert_eq!(v.len(), 42);
    /// }
    /// ```
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

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NeedlessPassByValue {
    #[allow(clippy::too_many_lines)]
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        body: &'tcx Body,
        span: Span,
        hir_id: HirId,
    ) {
        if in_macro_or_desugar(span) {
            return;
        }

        match kind {
            FnKind::ItemFn(.., header, _, attrs) => {
                if header.abi != Abi::Rust || requires_exact_signature(attrs) {
                    return;
                }
            },
            FnKind::Method(..) => (),
            _ => return,
        }

        // Exclude non-inherent impls
        if let Some(Node::Item(item)) = cx.tcx.hir().find(cx.tcx.hir().get_parent_node(hir_id)) {
            if matches!(item.node, ItemKind::Impl(_, _, _, _, Some(_), _, _) |
                ItemKind::Trait(..))
            {
                return;
            }
        }

        // Allow `Borrow` or functions to be taken by value
        let borrow_trait = need!(get_trait_def_id(cx, &paths::BORROW_TRAIT));
        let whitelisted_traits = [
            need!(cx.tcx.lang_items().fn_trait()),
            need!(cx.tcx.lang_items().fn_once_trait()),
            need!(cx.tcx.lang_items().fn_mut_trait()),
            need!(get_trait_def_id(cx, &paths::RANGE_ARGUMENT_TRAIT)),
        ];

        let sized_trait = need!(cx.tcx.lang_items().sized_trait());

        let fn_def_id = cx.tcx.hir().local_def_id(hir_id);

        let preds = traits::elaborate_predicates(cx.tcx, cx.param_env.caller_bounds.to_vec())
            .filter(|p| !p.is_global())
            .filter_map(|pred| {
                if let ty::Predicate::Trait(poly_trait_ref) = pred {
                    if poly_trait_ref.def_id() == sized_trait || poly_trait_ref.skip_binder().has_escaping_bound_vars()
                    {
                        return None;
                    }
                    Some(poly_trait_ref)
                } else {
                    None
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
            let mut ctx = MovedVariablesCtxt::new(cx);
            let region_scope_tree = &cx.tcx.region_scope_tree(fn_def_id);
            euv::ExprUseVisitor::new(
                &mut ctx,
                cx.tcx,
                fn_def_id,
                cx.param_env,
                region_scope_tree,
                cx.tables,
                None,
            )
            .consume_body(body);
            ctx
        };

        let fn_sig = cx.tcx.fn_sig(fn_def_id);
        let fn_sig = cx.tcx.erase_late_bound_regions(&fn_sig);

        for (idx, ((input, &ty), arg)) in decl.inputs.iter().zip(fn_sig.inputs()).zip(&body.arguments).enumerate() {
            // All spans generated from a proc-macro invocation are the same...
            if span == input.span {
                return;
            }

            // Ignore `self`s.
            if idx == 0 {
                if let PatKind::Binding(.., ident, _) = arg.pat.node {
                    if ident.as_str() == "self" {
                        continue;
                    }
                }
            }

            //
            // * Exclude a type that is specifically bounded by `Borrow`.
            // * Exclude a type whose reference also fulfills its bound. (e.g., `std::convert::AsRef`,
            //   `serde::Serialize`)
            let (implements_borrow_trait, all_borrowable_trait) = {
                let preds = preds
                    .iter()
                    .filter(|t| t.skip_binder().self_ty() == ty)
                    .collect::<Vec<_>>();

                (
                    preds.iter().any(|t| t.def_id() == borrow_trait),
                    !preds.is_empty()
                        && preds.iter().all(|t| {
                            let ty_params = &t
                                .skip_binder()
                                .trait_ref
                                .substs
                                .iter()
                                .skip(1)
                                .cloned()
                                .collect::<Vec<_>>();
                            implements_trait(cx, cx.tcx.mk_imm_ref(&RegionKind::ReEmpty, ty), t.def_id(), ty_params)
                        }),
                )
            };

            if_chain! {
                if !is_self(arg);
                if !ty.is_mutable_ptr();
                if !is_copy(cx, ty);
                if !whitelisted_traits.iter().any(|&t| implements_trait(cx, ty, t, &[]));
                if !implements_borrow_trait;
                if !all_borrowable_trait;

                if let PatKind::Binding(mode, canonical_id, ..) = arg.pat.node;
                if !moved_vars.contains(&canonical_id);
                then {
                    if mode == BindingAnnotation::Mutable || mode == BindingAnnotation::RefMut {
                        continue;
                    }

                    // Dereference suggestion
                    let sugg = |db: &mut DiagnosticBuilder<'_>| {
                        if let ty::Adt(def, ..) = ty.sty {
                            if let Some(span) = cx.tcx.hir().span_if_local(def.did) {
                                if cx.param_env.can_type_implement_copy(cx.tcx, ty).is_ok() {
                                    db.span_help(span, "consider marking this type as Copy");
                                }
                            }
                        }

                        let deref_span = spans_need_deref.get(&canonical_id);
                        if_chain! {
                            if match_type(cx, ty, &paths::VEC);
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_owned()")]);
                            if let TyKind::Path(QPath::Resolved(_, ref path)) = input.node;
                            if let Some(elem_ty) = path.segments.iter()
                                .find(|seg| seg.ident.name == sym!(Vec))
                                .and_then(|ps| ps.args.as_ref())
                                .map(|params| params.args.iter().find_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                }).unwrap());
                            then {
                                let slice_ty = format!("&[{}]", snippet(cx, elem_ty.span, "_"));
                                db.span_suggestion(
                                    input.span,
                                    "consider changing the type to",
                                    slice_ty,
                                    Applicability::Unspecified,
                                );

                                for (span, suggestion) in clone_spans {
                                    db.span_suggestion(
                                        span,
                                        &snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{}` to", x)),
                                            ),
                                        suggestion.into(),
                                        Applicability::Unspecified,
                                    );
                                }

                                // cannot be destructured, no need for `*` suggestion
                                assert!(deref_span.is_none());
                                return;
                            }
                        }

                        if match_type(cx, ty, &paths::STRING) {
                            if let Some(clone_spans) =
                                get_spans(cx, Some(body.id()), idx, &[("clone", ".to_string()"), ("as_str", "")]) {
                                db.span_suggestion(
                                    input.span,
                                    "consider changing the type to",
                                    "&str".to_string(),
                                    Applicability::Unspecified,
                                );

                                for (span, suggestion) in clone_spans {
                                    db.span_suggestion(
                                        span,
                                        &snippet_opt(cx, span)
                                            .map_or(
                                                "change the call to".into(),
                                                |x| Cow::from(format!("change `{}` to", x))
                                            ),
                                        suggestion.into(),
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
                                    .cloned()
                                    .map(|span| (span, format!("*{}", snippet(cx, span, "<expr>")))),
                            );
                            spans.sort_by_key(|&(span, _)| span);
                        }
                        multispan_sugg(db, "consider taking a reference instead".to_string(), spans);
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
        [sym!(proc_macro), sym!(proc_macro_attribute), sym!(proc_macro_derive)]
            .iter()
            .any(|&allow| attr.check_name(allow))
    })
}

struct MovedVariablesCtxt<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,
    moved_vars: FxHashSet<HirId>,
    /// Spans which need to be prefixed with `*` for dereferencing the
    /// suggested additional reference.
    spans_need_deref: FxHashMap<HirId, FxHashSet<Span>>,
}

impl<'a, 'tcx> MovedVariablesCtxt<'a, 'tcx> {
    fn new(cx: &'a LateContext<'a, 'tcx>) -> Self {
        Self {
            cx,
            moved_vars: FxHashSet::default(),
            spans_need_deref: FxHashMap::default(),
        }
    }

    fn move_common(&mut self, _consume_id: HirId, _span: Span, cmt: &mc::cmt_<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if let mc::Categorization::Local(vid) = cmt.cat {
            self.moved_vars.insert(vid);
        }
    }

    fn non_moving_pat(&mut self, matched_pat: &Pat, cmt: &mc::cmt_<'tcx>) {
        let cmt = unwrap_downcast_or_interior(cmt);

        if let mc::Categorization::Local(vid) = cmt.cat {
            let mut id = matched_pat.hir_id;
            loop {
                let parent = self.cx.tcx.hir().get_parent_node(id);
                if id == parent {
                    // no parent
                    return;
                }
                id = parent;

                if let Some(node) = self.cx.tcx.hir().find(id) {
                    match node {
                        Node::Expr(e) => {
                            // `match` and `if let`
                            if let ExprKind::Match(ref c, ..) = e.node {
                                self.spans_need_deref
                                    .entry(vid)
                                    .or_insert_with(FxHashSet::default)
                                    .insert(c.span);
                            }
                        },

                        Node::Stmt(s) => {
                            // `let <pat> = x;`
                            if_chain! {
                                if let StmtKind::Local(ref local) = s.node;
                                then {
                                    self.spans_need_deref
                                        .entry(vid)
                                        .or_insert_with(FxHashSet::default)
                                        .insert(local.init
                                            .as_ref()
                                            .map(|e| e.span)
                                            .expect("`let` stmt without init aren't caught by match_pat"));
                                }
                            }
                        },

                        _ => {},
                    }
                }
            }
        }
    }
}

impl<'a, 'tcx> euv::Delegate<'tcx> for MovedVariablesCtxt<'a, 'tcx> {
    fn consume(&mut self, consume_id: HirId, consume_span: Span, cmt: &mc::cmt_<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_id, consume_span, cmt);
        }
    }

    fn matched_pat(&mut self, matched_pat: &Pat, cmt: &mc::cmt_<'tcx>, mode: euv::MatchMode) {
        if let euv::MatchMode::MovingMatch = mode {
            self.move_common(matched_pat.hir_id, matched_pat.span, cmt);
        } else {
            self.non_moving_pat(matched_pat, cmt);
        }
    }

    fn consume_pat(&mut self, consume_pat: &Pat, cmt: &mc::cmt_<'tcx>, mode: euv::ConsumeMode) {
        if let euv::ConsumeMode::Move(_) = mode {
            self.move_common(consume_pat.hir_id, consume_pat.span, cmt);
        }
    }

    fn borrow(
        &mut self,
        _: HirId,
        _: Span,
        _: &mc::cmt_<'tcx>,
        _: ty::Region<'_>,
        _: ty::BorrowKind,
        _: euv::LoanCause,
    ) {
    }

    fn mutate(&mut self, _: HirId, _: Span, _: &mc::cmt_<'tcx>, _: euv::MutateMode) {}

    fn decl_without_init(&mut self, _: HirId, _: Span) {}
}

fn unwrap_downcast_or_interior<'a, 'tcx>(mut cmt: &'a mc::cmt_<'tcx>) -> mc::cmt_<'tcx> {
    loop {
        match cmt.cat {
            mc::Categorization::Downcast(ref c, _) | mc::Categorization::Interior(ref c, _) => {
                cmt = c;
            },
            _ => return (*cmt).clone(),
        }
    }
}
