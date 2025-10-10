use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::mir::{PossibleBorrowerMap, enclosing_mir, expr_local, local_assignments, used_exactly_once};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::snippet_with_context;
use clippy_utils::ty::{implements_trait, is_copy};
use clippy_utils::{DefinedTy, ExprUseNode, expr_use_ctxt, peel_n_hir_expr_refs};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{Body, Expr, ExprKind, Mutability, Path, QPath};
use rustc_index::bit_set::DenseBitSet;
use rustc_infer::infer::TyCtxtInferExt;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::{Rvalue, StatementKind};
use rustc_middle::ty::{
    self, ClauseKind, EarlyBinder, FnSig, GenericArg, GenericArgKind, ParamTy, ProjectionPredicate, Ty,
};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{Obligation, ObligationCause};
use std::collections::VecDeque;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for borrow operations (`&`) that are used as a generic argument to a
    /// function when the borrowed value could be used.
    ///
    /// ### Why is this bad?
    /// Suggests that the receiver of the expression borrows
    /// the expression.
    ///
    /// ### Known problems
    /// The lint cannot tell when the implementation of a trait
    /// for `&T` and `T` do different things. Removing a borrow
    /// in such a case can change the semantics of the code.
    ///
    /// ### Example
    /// ```no_run
    /// fn f(_: impl AsRef<str>) {}
    ///
    /// let x = "foo";
    /// f(&x);
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// fn f(_: impl AsRef<str>) {}
    ///
    /// let x = "foo";
    /// f(x);
    /// ```
    #[clippy::version = "1.74.0"]
    pub NEEDLESS_BORROWS_FOR_GENERIC_ARGS,
    style,
    "taking a reference that is going to be automatically dereferenced"
}

pub struct NeedlessBorrowsForGenericArgs<'tcx> {
    /// Stack of (body owner, `PossibleBorrowerMap`) pairs. Used by
    /// [`needless_borrow_count`] to determine when a borrowed expression can instead
    /// be moved.
    possible_borrowers: Vec<(LocalDefId, PossibleBorrowerMap<'tcx, 'tcx>)>,

    // `IntoIterator` for arrays requires Rust 1.53.
    msrv: Msrv,
}
impl_lint_pass!(NeedlessBorrowsForGenericArgs<'_> => [NEEDLESS_BORROWS_FOR_GENERIC_ARGS]);

impl NeedlessBorrowsForGenericArgs<'_> {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            possible_borrowers: Vec::new(),
            msrv: conf.msrv,
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NeedlessBorrowsForGenericArgs<'tcx> {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if matches!(expr.kind, ExprKind::AddrOf(..))
            && !expr.span.from_expansion()
            && let use_cx = expr_use_ctxt(cx, expr)
            && use_cx.same_ctxt
            && !use_cx.is_ty_unified
            && let use_node = use_cx.use_node(cx)
            && let Some(DefinedTy::Mir { def_site_def_id: _, ty }) = use_node.defined_ty(cx)
            && let ty::Param(param_ty) = *ty.skip_binder().kind()
            && let Some((hir_id, fn_id, i)) = match use_node {
                ExprUseNode::MethodArg(_, _, 0) => None,
                ExprUseNode::MethodArg(hir_id, None, i) => cx
                    .typeck_results()
                    .type_dependent_def_id(hir_id)
                    .map(|id| (hir_id, id, i)),
                ExprUseNode::FnArg(
                    &Expr {
                        kind: ExprKind::Path(ref p),
                        hir_id,
                        ..
                    },
                    i,
                ) if !path_has_args(p) => match cx.typeck_results().qpath_res(p, hir_id) {
                    Res::Def(DefKind::Fn | DefKind::Ctor(..) | DefKind::AssocFn, id) => Some((hir_id, id, i)),
                    _ => None,
                },
                _ => None,
            }
            && let count = needless_borrow_count(
                cx,
                &mut self.possible_borrowers,
                fn_id,
                cx.typeck_results().node_args(hir_id),
                i,
                param_ty,
                expr,
                self.msrv,
            )
            && count != 0
        {
            span_lint_and_then(
                cx,
                NEEDLESS_BORROWS_FOR_GENERIC_ARGS,
                expr.span,
                "the borrowed expression implements the required traits",
                |diag| {
                    let mut app = Applicability::MachineApplicable;
                    let snip_span = peel_n_hir_expr_refs(expr, count).0.span;
                    let snip = snippet_with_context(cx, snip_span, expr.span.ctxt(), "..", &mut app).0;
                    diag.span_suggestion(expr.span, "change this to", snip.into_owned(), app);
                },
            );
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, body: &Body<'_>) {
        if self
            .possible_borrowers
            .last()
            .is_some_and(|&(local_def_id, _)| local_def_id == cx.tcx.hir_body_owner_def_id(body.id()))
        {
            self.possible_borrowers.pop();
        }
    }
}

fn path_has_args(p: &QPath<'_>) -> bool {
    match *p {
        QPath::Resolved(_, Path { segments: [.., s], .. }) | QPath::TypeRelative(_, s) => s.args.is_some(),
        _ => false,
    }
}

/// Checks for the number of borrow expressions which can be removed from the given expression
/// where the expression is used as an argument to a function expecting a generic type.
///
/// The following constraints will be checked:
/// * The borrowed expression meets all the generic type's constraints.
/// * The generic type appears only once in the functions signature.
/// * The borrowed value is:
///   - `Copy` itself, or
///   - the only use of a mutable reference, or
///   - not a variable (created by a function call)
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
fn needless_borrow_count<'tcx>(
    cx: &LateContext<'tcx>,
    possible_borrowers: &mut Vec<(LocalDefId, PossibleBorrowerMap<'tcx, 'tcx>)>,
    fn_id: DefId,
    callee_args: ty::GenericArgsRef<'tcx>,
    arg_index: usize,
    param_ty: ParamTy,
    mut expr: &Expr<'tcx>,
    msrv: Msrv,
) -> usize {
    let destruct_trait_def_id = cx.tcx.lang_items().destruct_trait();
    let sized_trait_def_id = cx.tcx.lang_items().sized_trait();
    let meta_sized_trait_def_id = cx.tcx.lang_items().meta_sized_trait();
    let drop_trait_def_id = cx.tcx.lang_items().drop_trait();

    let fn_sig = cx.tcx.fn_sig(fn_id).instantiate_identity().skip_binder();
    let predicates = cx.tcx.param_env(fn_id).caller_bounds();
    let projection_predicates = predicates
        .iter()
        .filter_map(|predicate| {
            if let ClauseKind::Projection(projection_predicate) = predicate.kind().skip_binder() {
                Some(projection_predicate)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut trait_with_ref_mut_self_method = false;

    // If no traits were found, or only the `Destruct`, `Sized`, or `Any` traits were found, return.
    if predicates
        .iter()
        .filter_map(|predicate| {
            if let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder()
                && trait_predicate.trait_ref.self_ty() == param_ty.to_ty(cx.tcx)
            {
                Some(trait_predicate.trait_ref.def_id)
            } else {
                None
            }
        })
        .inspect(|trait_def_id| {
            trait_with_ref_mut_self_method |= has_ref_mut_self_method(cx, *trait_def_id);
        })
        .all(|trait_def_id| {
            Some(trait_def_id) == destruct_trait_def_id
                || Some(trait_def_id) == sized_trait_def_id
                || Some(trait_def_id) == meta_sized_trait_def_id
                || cx.tcx.is_diagnostic_item(sym::Any, trait_def_id)
        })
    {
        return 0;
    }

    // See:
    // - https://github.com/rust-lang/rust-clippy/pull/9674#issuecomment-1289294201
    // - https://github.com/rust-lang/rust-clippy/pull/9674#issuecomment-1292225232
    if projection_predicates
        .iter()
        .any(|projection_predicate| is_mixed_projection_predicate(cx, fn_id, projection_predicate))
    {
        return 0;
    }

    // `args_with_referent_ty` can be constructed outside of `check_referent` because the same
    // elements are modified each time `check_referent` is called.
    let mut args_with_referent_ty = callee_args.to_vec();

    let mut check_reference_and_referent = |reference: &Expr<'tcx>, referent: &Expr<'tcx>| {
        if let ExprKind::Field(base, _) = &referent.kind
            && let base_ty = cx.typeck_results().expr_ty(base)
            && drop_trait_def_id.is_some_and(|id| implements_trait(cx, base_ty, id, &[]))
        {
            return false;
        }

        let referent_ty = cx.typeck_results().expr_ty(referent);

        if !(is_copy(cx, referent_ty)
            || referent_ty.is_ref() && referent_used_exactly_once(cx, possible_borrowers, reference)
            || matches!(referent.kind, ExprKind::Call(..) | ExprKind::MethodCall(..)))
        {
            return false;
        }

        // https://github.com/rust-lang/rust-clippy/pull/9136#pullrequestreview-1037379321
        if trait_with_ref_mut_self_method && !matches!(referent_ty.kind(), ty::Ref(_, _, Mutability::Mut)) {
            return false;
        }

        if !replace_types(
            cx,
            param_ty,
            referent_ty,
            fn_sig,
            arg_index,
            &projection_predicates,
            &mut args_with_referent_ty,
        ) {
            return false;
        }

        predicates.iter().all(|predicate| {
            if let ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder()
                && cx
                    .tcx
                    .is_diagnostic_item(sym::IntoIterator, trait_predicate.trait_ref.def_id)
                && let ty::Param(param_ty) = trait_predicate.self_ty().kind()
                && let GenericArgKind::Type(ty) = args_with_referent_ty[param_ty.index as usize].kind()
                && ty.is_array()
                && !msrv.meets(cx, msrvs::ARRAY_INTO_ITERATOR)
            {
                return false;
            }

            let predicate = EarlyBinder::bind(predicate).instantiate(cx.tcx, &args_with_referent_ty[..]);
            let obligation = Obligation::new(cx.tcx, ObligationCause::dummy(), cx.param_env, predicate);
            let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
            infcx.predicate_must_hold_modulo_regions(&obligation)
        })
    };

    let mut count = 0;
    while let ExprKind::AddrOf(_, _, referent) = expr.kind {
        if !check_reference_and_referent(expr, referent) {
            break;
        }
        expr = referent;
        count += 1;
    }
    count
}

fn has_ref_mut_self_method(cx: &LateContext<'_>, trait_def_id: DefId) -> bool {
    cx.tcx
        .associated_items(trait_def_id)
        .in_definition_order()
        .any(|assoc_item| {
            if assoc_item.is_method() {
                let self_ty = cx
                    .tcx
                    .fn_sig(assoc_item.def_id)
                    .instantiate_identity()
                    .skip_binder()
                    .inputs()[0];
                matches!(self_ty.kind(), ty::Ref(_, _, Mutability::Mut))
            } else {
                false
            }
        })
}

fn is_mixed_projection_predicate<'tcx>(
    cx: &LateContext<'tcx>,
    callee_def_id: DefId,
    projection_predicate: &ProjectionPredicate<'tcx>,
) -> bool {
    let generics = cx.tcx.generics_of(callee_def_id);
    // The predicate requires the projected type to equal a type parameter from the parent context.
    if let Some(term_ty) = projection_predicate.term.as_type()
        && let ty::Param(term_param_ty) = term_ty.kind()
        && (term_param_ty.index as usize) < generics.parent_count
    {
        // The inner-most self type is a type parameter from the current function.
        let mut projection_term = projection_predicate.projection_term;
        loop {
            match *projection_term.self_ty().kind() {
                ty::Alias(ty::Projection, inner_projection_ty) => {
                    projection_term = inner_projection_ty.into();
                },
                ty::Param(param_ty) => {
                    return (param_ty.index as usize) >= generics.parent_count;
                },
                _ => {
                    return false;
                },
            }
        }
    } else {
        false
    }
}

fn referent_used_exactly_once<'tcx>(
    cx: &LateContext<'tcx>,
    possible_borrowers: &mut Vec<(LocalDefId, PossibleBorrowerMap<'tcx, 'tcx>)>,
    reference: &Expr<'tcx>,
) -> bool {
    if let Some(mir) = enclosing_mir(cx.tcx, reference.hir_id)
        && let Some(local) = expr_local(cx.tcx, reference)
        && let [location] = *local_assignments(mir, local).as_slice()
        && let block_data = &mir.basic_blocks[location.block]
        && let Some(statement) = block_data.statements.get(location.statement_index)
        && let StatementKind::Assign(box (_, Rvalue::Ref(_, _, place))) = statement.kind
        && !place.is_indirect_first_projection()
    {
        let body_owner_local_def_id = cx.tcx.hir_enclosing_body_owner(reference.hir_id);
        if possible_borrowers
            .last()
            .is_none_or(|&(local_def_id, _)| local_def_id != body_owner_local_def_id)
        {
            possible_borrowers.push((body_owner_local_def_id, PossibleBorrowerMap::new(cx, mir)));
        }
        let possible_borrower = &mut possible_borrowers.last_mut().unwrap().1;
        // If `only_borrowers` were used here, the `copyable_iterator::warn` test would fail. The reason is
        // that `PossibleBorrowerVisitor::visit_terminator` considers `place.local` a possible borrower of
        // itself. See the comment in that method for an explanation as to why.
        possible_borrower.bounded_borrowers(&[local], &[local, place.local], place.local, location)
            && used_exactly_once(mir, place.local).unwrap_or(false)
    } else {
        false
    }
}

// Iteratively replaces `param_ty` with `new_ty` in `args`, and similarly for each resulting
// projected type that is a type parameter. Returns `false` if replacing the types would have an
// effect on the function signature beyond substituting `new_ty` for `param_ty`.
// See: https://github.com/rust-lang/rust-clippy/pull/9136#discussion_r927212757
fn replace_types<'tcx>(
    cx: &LateContext<'tcx>,
    param_ty: ParamTy,
    new_ty: Ty<'tcx>,
    fn_sig: FnSig<'tcx>,
    arg_index: usize,
    projection_predicates: &[ProjectionPredicate<'tcx>],
    args: &mut [GenericArg<'tcx>],
) -> bool {
    let mut replaced = DenseBitSet::new_empty(args.len());

    let mut deque = VecDeque::with_capacity(args.len());
    deque.push_back((param_ty, new_ty));

    while let Some((param_ty, new_ty)) = deque.pop_front() {
        // If `replaced.is_empty()`, then `param_ty` and `new_ty` are those initially passed in.
        if !fn_sig
            .inputs_and_output
            .iter()
            .enumerate()
            .all(|(i, ty)| (replaced.is_empty() && i == arg_index) || !ty.contains(param_ty.to_ty(cx.tcx)))
        {
            return false;
        }

        args[param_ty.index as usize] = GenericArg::from(new_ty);

        // The `replaced.insert(...)` check provides some protection against infinite loops.
        if replaced.insert(param_ty.index) {
            for projection_predicate in projection_predicates {
                if projection_predicate.projection_term.self_ty() == param_ty.to_ty(cx.tcx)
                    && let Some(term_ty) = projection_predicate.term.as_type()
                    && let ty::Param(term_param_ty) = term_ty.kind()
                {
                    let projection = projection_predicate
                        .projection_term
                        .with_replaced_self_ty(cx.tcx, new_ty)
                        .expect_ty(cx.tcx)
                        .to_ty(cx.tcx);

                    if let Ok(projected_ty) = cx.tcx.try_normalize_erasing_regions(cx.typing_env(), projection)
                        && args[term_param_ty.index as usize] != GenericArg::from(projected_ty)
                    {
                        deque.push_back((*term_param_ty, projected_ty));
                    }
                }
            }
        }
    }

    true
}
