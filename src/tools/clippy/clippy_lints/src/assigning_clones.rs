use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::mir::{PossibleBorrowerMap, enclosing_mir};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_diag_trait_item, is_in_test, last_path_segment, local_is_initialized, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir;
use rustc_middle::ty::{self, Instance, Mutability};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;
use rustc_span::{Span, SyntaxContext};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for code like `foo = bar.clone();`
    ///
    /// ### Why is this bad?
    /// Custom `Clone::clone_from()` or `ToOwned::clone_into` implementations allow the objects
    /// to share resources and therefore avoid allocations.
    ///
    /// ### Example
    /// ```rust
    /// struct Thing;
    ///
    /// impl Clone for Thing {
    ///     fn clone(&self) -> Self { todo!() }
    ///     fn clone_from(&mut self, other: &Self) { todo!() }
    /// }
    ///
    /// pub fn assign_to_ref(a: &mut Thing, b: Thing) {
    ///     *a = b.clone();
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct Thing;
    ///
    /// impl Clone for Thing {
    ///     fn clone(&self) -> Self { todo!() }
    ///     fn clone_from(&mut self, other: &Self) { todo!() }
    /// }
    ///
    /// pub fn assign_to_ref(a: &mut Thing, b: Thing) {
    ///     a.clone_from(&b);
    /// }
    /// ```
    #[clippy::version = "1.78.0"]
    pub ASSIGNING_CLONES,
    pedantic,
    "assigning the result of cloning may be inefficient"
}

pub struct AssigningClones {
    msrv: Msrv,
}

impl AssigningClones {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}

impl_lint_pass!(AssigningClones => [ASSIGNING_CLONES]);

impl<'tcx> LateLintPass<'tcx> for AssigningClones {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Assign(lhs, rhs, _) = e.kind
            && let typeck = cx.typeck_results()
            && let (call_kind, fn_name, fn_id, fn_arg, fn_gen_args) = match rhs.kind {
                ExprKind::Call(f, [arg])
                    if let ExprKind::Path(fn_path) = &f.kind
                        && let Some(id) = typeck.qpath_res(fn_path, f.hir_id).opt_def_id() =>
                {
                    (CallKind::Ufcs, last_path_segment(fn_path).ident.name, id, arg, typeck.node_args(f.hir_id))
                },
                ExprKind::MethodCall(name, recv, [], _) if let Some(id) = typeck.type_dependent_def_id(rhs.hir_id) => {
                    (CallKind::Method, name.ident.name, id, recv, typeck.node_args(rhs.hir_id))
                },
                _ => return,
            }
            && let ctxt = e.span.ctxt()
            // Don't lint in macros.
            && ctxt.is_root()
            && let which_trait = match fn_name {
                sym::clone if is_diag_trait_item(cx, fn_id, sym::Clone) => CloneTrait::Clone,
                _ if fn_name.as_str() == "to_owned"
                    && is_diag_trait_item(cx, fn_id, sym::ToOwned)
                    && self.msrv.meets(cx, msrvs::CLONE_INTO) =>
                {
                    CloneTrait::ToOwned
                },
                _ => return,
            }
            && let Ok(Some(resolved_fn)) = Instance::try_resolve(cx.tcx, cx.typing_env(), fn_id, fn_gen_args)
            // TODO: This check currently bails if the local variable has no initializer.
            // That is overly conservative - the lint should fire even if there was no initializer,
            // but the variable has been initialized before `lhs` was evaluated.
            && path_to_local(lhs).is_none_or(|lhs| local_is_initialized(cx, lhs))
            && let Some(resolved_impl) = cx.tcx.impl_of_method(resolved_fn.def_id())
            // Derived forms don't implement `clone_from`/`clone_into`.
            // See https://github.com/rust-lang/rust/pull/98445#issuecomment-1190681305
            && !cx.tcx.is_builtin_derived(resolved_impl)
            // Don't suggest calling a function we're implementing.
            && resolved_impl.as_local().is_none_or(|block_id| {
                cx.tcx.hir_parent_owner_iter(e.hir_id).all(|(id, _)| id.def_id != block_id)
            })
            && let resolved_assoc_items = cx.tcx.associated_items(resolved_impl)
            // Only suggest if `clone_from`/`clone_into` is explicitly implemented
            && resolved_assoc_items.in_definition_order().any(|assoc|
                match which_trait {
                    CloneTrait::Clone => assoc.name() == sym::clone_from,
                    CloneTrait::ToOwned => assoc.name().as_str() == "clone_into",
                }
            )
            && !clone_source_borrows_from_dest(cx, lhs, rhs.span)
            && !is_in_test(cx.tcx, e.hir_id)
        {
            span_lint_and_then(
                cx,
                ASSIGNING_CLONES,
                e.span,
                match which_trait {
                    CloneTrait::Clone => "assigning the result of `Clone::clone()` may be inefficient",
                    CloneTrait::ToOwned => "assigning the result of `ToOwned::to_owned()` may be inefficient",
                },
                |diag| {
                    let mut app = Applicability::Unspecified;
                    diag.span_suggestion(
                        e.span,
                        match which_trait {
                            CloneTrait::Clone => "use `clone_from()`",
                            CloneTrait::ToOwned => "use `clone_into()`",
                        },
                        build_sugg(cx, ctxt, lhs, fn_arg, which_trait, call_kind, &mut app),
                        app,
                    );
                },
            );
        }
    }
}

/// Checks if the data being cloned borrows from the place that is being assigned to:
///
/// ```
/// let mut s = String::new();
/// let s2 = &s;
/// s = s2.to_owned();
/// ```
///
/// This cannot be written `s2.clone_into(&mut s)` because it has conflicting borrows.
fn clone_source_borrows_from_dest(cx: &LateContext<'_>, lhs: &Expr<'_>, call_span: Span) -> bool {
    let Some(mir) = enclosing_mir(cx.tcx, lhs.hir_id) else {
        return false;
    };
    let PossibleBorrowerMap { map: borrow_map, .. } = PossibleBorrowerMap::new(cx, mir);

    // The operation `dest = src.to_owned()` in MIR is split up across 3 blocks *if* the type has `Drop`
    // code. For types that don't, the second basic block is simply skipped.
    // For the doc example above that would be roughly:
    //
    // bb0:
    //  s2 = &s
    //  s_temp = ToOwned::to_owned(move s2) -> bb1
    //
    // bb1:
    //  drop(s) -> bb2  // drop the old string
    //
    // bb2:
    //  s = s_temp
    if let Some(terminator) = mir.basic_blocks.iter()
            .map(mir::BasicBlockData::terminator)
            .find(|term| term.source_info.span == call_span)
        && let mir::TerminatorKind::Call { ref args, target: Some(assign_bb), .. } = terminator.kind
        && let [source] = &**args
        && let mir::Operand::Move(source) = &source.node
        && let assign_bb = &mir.basic_blocks[assign_bb]
        && let assign_bb = match assign_bb.terminator().kind {
            // Skip the drop of the assignment's destination.
            mir::TerminatorKind::Drop { target, .. } => &mir.basic_blocks[target],
            _ => assign_bb,
        }
        // Skip any storage statements as they are just noise
        && let Some(assignment) = assign_bb.statements
            .iter()
            .find(|stmt| {
                !matches!(stmt.kind, mir::StatementKind::StorageDead(_) | mir::StatementKind::StorageLive(_))
            })
        && let mir::StatementKind::Assign(box (borrowed, _)) = &assignment.kind
        && let Some(borrowers) = borrow_map.get(&borrowed.local)
    {
        borrowers.contains(source.local)
    } else {
        false
    }
}

#[derive(Clone, Copy)]
enum CloneTrait {
    Clone,
    ToOwned,
}

#[derive(Copy, Clone)]
enum CallKind {
    Ufcs,
    Method,
}

fn build_sugg<'tcx>(
    cx: &LateContext<'tcx>,
    ctxt: SyntaxContext,
    lhs: &'tcx Expr<'_>,
    fn_arg: &'tcx Expr<'_>,
    which_trait: CloneTrait,
    call_kind: CallKind,
    app: &mut Applicability,
) -> String {
    match which_trait {
        CloneTrait::Clone => {
            match call_kind {
                CallKind::Method => {
                    let receiver_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                        // If `ref_expr` is a reference, we can remove the dereference operator (`*`) to make
                        // the generated code a bit simpler. In other cases, we don't do this special case, to avoid
                        // having to deal with Deref (https://github.com/rust-lang/rust-clippy/issues/12437).

                        let ty = cx.typeck_results().expr_ty(ref_expr);
                        if ty.is_ref() {
                            // Apply special case, remove `*`
                            // `*lhs = self_expr.clone();` -> `lhs.clone_from(self_expr)`
                            Sugg::hir_with_applicability(cx, ref_expr, "_", app)
                        } else {
                            // Keep the original lhs
                            // `*lhs = self_expr.clone();` -> `(*lhs).clone_from(self_expr)`
                            Sugg::hir_with_applicability(cx, lhs, "_", app)
                        }
                    } else {
                        // Keep the original lhs
                        // `lhs = self_expr.clone();` -> `lhs.clone_from(self_expr)`
                        Sugg::hir_with_applicability(cx, lhs, "_", app)
                    }
                    .maybe_par();

                    // Determine whether we need to reference the argument to clone_from().
                    let clone_receiver_type = cx.typeck_results().expr_ty(fn_arg);
                    let clone_receiver_adj_type = cx.typeck_results().expr_ty_adjusted(fn_arg);
                    let mut arg_sugg = Sugg::hir_with_context(cx, fn_arg, ctxt, "_", app);
                    if clone_receiver_type != clone_receiver_adj_type {
                        // The receiver may have been a value type, so we need to add an `&` to
                        // be sure the argument to clone_from will be a reference.
                        arg_sugg = arg_sugg.addr();
                    }

                    format!("{receiver_sugg}.clone_from({arg_sugg})")
                },
                CallKind::Ufcs => {
                    let self_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                        // See special case of removing `*` in method handling above
                        let ty = cx.typeck_results().expr_ty(ref_expr);
                        if ty.is_ref() {
                            // `*lhs = Clone::clone(self_expr);` -> `Clone::clone_from(lhs, self_expr)`
                            Sugg::hir_with_applicability(cx, ref_expr, "_", app)
                        } else {
                            // `*lhs = Clone::clone(self_expr);` -> `Clone::clone_from(&mut *lhs, self_expr)`
                            // mut_addr_deref is used to avoid unnecessary parentheses around `*lhs`
                            Sugg::hir_with_applicability(cx, ref_expr, "_", app).mut_addr_deref()
                        }
                    } else {
                        // `lhs = Clone::clone(self_expr);` -> `Clone::clone_from(&mut lhs, self_expr)`
                        Sugg::hir_with_applicability(cx, lhs, "_", app).mut_addr()
                    };
                    // The RHS had to be exactly correct before the call, there is no auto-deref for function calls.
                    let rhs_sugg = Sugg::hir_with_context(cx, fn_arg, ctxt, "_", app);

                    format!("Clone::clone_from({self_sugg}, {rhs_sugg})")
                },
            }
        },
        CloneTrait::ToOwned => {
            let rhs_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                // `*lhs = rhs.to_owned()` -> `rhs.clone_into(lhs)`
                // `*lhs = ToOwned::to_owned(rhs)` -> `ToOwned::clone_into(rhs, lhs)`
                let sugg = Sugg::hir_with_applicability(cx, ref_expr, "_", app).maybe_par();
                let inner_type = cx.typeck_results().expr_ty(ref_expr);
                // If after unwrapping the dereference, the type is not a mutable reference, we add &mut to make it
                // deref to a mutable reference.
                if matches!(inner_type.kind(), ty::Ref(_, _, Mutability::Mut)) {
                    sugg
                } else {
                    sugg.mut_addr()
                }
            } else {
                // `lhs = rhs.to_owned()` -> `rhs.clone_into(&mut lhs)`
                // `lhs = ToOwned::to_owned(rhs)` -> `ToOwned::clone_into(rhs, &mut lhs)`
                Sugg::hir_with_applicability(cx, lhs, "_", app).maybe_par().mut_addr()
            };

            match call_kind {
                CallKind::Method => {
                    let receiver_sugg = Sugg::hir_with_context(cx, fn_arg, ctxt, "_", app);
                    format!("{receiver_sugg}.clone_into({rhs_sugg})")
                },
                CallKind::Ufcs => {
                    let self_sugg = Sugg::hir_with_context(cx, fn_arg, ctxt, "_", app);
                    format!("ToOwned::clone_into({self_sugg}, {rhs_sugg})")
                },
            }
        },
    }
}
