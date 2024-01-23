use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::HirNode;
use clippy_utils::sugg::Sugg;
use clippy_utils::{is_trait_method, path_to_local};
use rustc_errors::Applicability;
use rustc_hir::{self as hir, Expr, ExprKind, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, Instance, Mutability};
use rustc_session::declare_lint_pass;
use rustc_span::def_id::DefId;
use rustc_span::symbol::sym;
use rustc_span::ExpnKind;

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
    /// impl Clone for Thing {
    ///     fn clone(&self) -> Self { todo!() }
    ///     fn clone_from(&mut self, other: &Self) { todo!() }
    /// }
    ///
    /// pub fn assign_to_ref(a: &mut Thing, b: Thing) {
    ///     a.clone_from(&b);
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub ASSIGNING_CLONES,
    perf,
    "assigning the result of cloning may be inefficient"
}
declare_lint_pass!(AssigningClones => [ASSIGNING_CLONES]);

impl<'tcx> LateLintPass<'tcx> for AssigningClones {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, assign_expr: &'tcx hir::Expr<'_>) {
        // Do not fire the lint in macros
        let expn_data = assign_expr.span().ctxt().outer_expn_data();
        match expn_data.kind {
            ExpnKind::AstPass(_) | ExpnKind::Desugaring(_) | ExpnKind::Macro(..) => return,
            ExpnKind::Root => {},
        }

        let ExprKind::Assign(lhs, rhs, _span) = assign_expr.kind else {
            return;
        };

        let Some(call) = extract_call(cx, rhs) else {
            return;
        };

        if is_ok_to_suggest(cx, lhs, &call) {
            suggest(cx, assign_expr, lhs, &call);
        }
    }
}

// Try to resolve the call to `Clone::clone` or `ToOwned::to_owned`.
fn extract_call<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<CallCandidate<'tcx>> {
    let fn_def_id = clippy_utils::fn_def_id(cx, expr)?;

    // Fast paths to only check method calls without arguments or function calls with a single argument
    let (target, kind, resolved_method) = match expr.kind {
        ExprKind::MethodCall(path, receiver, [], _span) => {
            let args = cx.typeck_results().node_args(expr.hir_id);

            // If we could not resolve the method, don't apply the lint
            let Ok(Some(resolved_method)) = Instance::resolve(cx.tcx, cx.param_env, fn_def_id, args) else {
                return None;
            };
            if is_trait_method(cx, expr, sym::Clone) && path.ident.name == sym::clone {
                (TargetTrait::Clone, CallKind::MethodCall { receiver }, resolved_method)
            } else if is_trait_method(cx, expr, sym::ToOwned) && path.ident.name.as_str() == "to_owned" {
                (TargetTrait::ToOwned, CallKind::MethodCall { receiver }, resolved_method)
            } else {
                return None;
            }
        },
        ExprKind::Call(function, [arg]) => {
            let kind = cx.typeck_results().node_type(function.hir_id).kind();

            // If we could not resolve the method, don't apply the lint
            let Ok(Some(resolved_method)) = (match kind {
                ty::FnDef(_, args) => Instance::resolve(cx.tcx, cx.param_env, fn_def_id, args),
                _ => Ok(None),
            }) else {
                return None;
            };
            if cx.tcx.is_diagnostic_item(sym::to_owned_method, fn_def_id) {
                (
                    TargetTrait::ToOwned,
                    CallKind::FunctionCall { self_arg: arg },
                    resolved_method,
                )
            } else if let Some(trait_did) = cx.tcx.trait_of_item(fn_def_id)
                && cx.tcx.is_diagnostic_item(sym::Clone, trait_did)
            {
                (
                    TargetTrait::Clone,
                    CallKind::FunctionCall { self_arg: arg },
                    resolved_method,
                )
            } else {
                return None;
            }
        },
        _ => return None,
    };

    Some(CallCandidate {
        target,
        kind,
        method_def_id: resolved_method.def_id(),
    })
}

// Return true if we find that the called method has a custom implementation and isn't derived or
// provided by default by the corresponding trait.
fn is_ok_to_suggest<'tcx>(cx: &LateContext<'tcx>, lhs: &Expr<'tcx>, call: &CallCandidate<'tcx>) -> bool {
    // If the left-hand side is a local variable, it might be uninitialized at this point.
    // In that case we do not want to suggest the lint.
    if let Some(local) = path_to_local(lhs) {
        // TODO: This check currently bails if the local variable has no initializer.
        // That is overly conservative - the lint should fire even if there was no initializer,
        // but the variable has been initialized before `lhs` was evaluated.
        if let Some(Node::Local(local)) = cx.tcx.hir().parent_id_iter(local).next().map(|p| cx.tcx.hir_node(p))
            && local.init.is_none()
        {
            return false;
        }
    }

    let Some(impl_block) = cx.tcx.impl_of_method(call.method_def_id) else {
        return false;
    };

    // If the method implementation comes from #[derive(Clone)], then don't suggest the lint.
    // Automatically generated Clone impls do not currently override `clone_from`.
    // See e.g. https://github.com/rust-lang/rust/pull/98445#issuecomment-1190681305 for more details.
    if cx.tcx.is_builtin_derived(impl_block) {
        return false;
    }

    // Find the function for which we want to check that it is implemented.
    let provided_fn = match call.target {
        TargetTrait::Clone => cx.tcx.get_diagnostic_item(sym::Clone).and_then(|clone| {
            cx.tcx
                .provided_trait_methods(clone)
                .find(|item| item.name == sym::clone_from)
        }),
        TargetTrait::ToOwned => cx.tcx.get_diagnostic_item(sym::ToOwned).and_then(|to_owned| {
            cx.tcx
                .provided_trait_methods(to_owned)
                .find(|item| item.name.as_str() == "clone_into")
        }),
    };
    let Some(provided_fn) = provided_fn else {
        return false;
    };

    // Now take a look if the impl block defines an implementation for the method that we're interested
    // in. If not, then we're using a default implementation, which is not interesting, so we will
    // not suggest the lint.
    let implemented_fns = cx.tcx.impl_item_implementor_ids(impl_block);
    implemented_fns.contains_key(&provided_fn.def_id)
}

fn suggest<'tcx>(
    cx: &LateContext<'tcx>,
    assign_expr: &hir::Expr<'tcx>,
    lhs: &hir::Expr<'tcx>,
    call: &CallCandidate<'tcx>,
) {
    span_lint_and_then(cx, ASSIGNING_CLONES, assign_expr.span, call.message(), |diag| {
        let mut applicability = Applicability::MachineApplicable;

        diag.span_suggestion(
            assign_expr.span,
            call.suggestion_msg(),
            call.suggested_replacement(cx, lhs, &mut applicability),
            applicability,
        );
    });
}

#[derive(Copy, Clone, Debug)]
enum CallKind<'tcx> {
    MethodCall { receiver: &'tcx Expr<'tcx> },
    FunctionCall { self_arg: &'tcx Expr<'tcx> },
}

#[derive(Copy, Clone, Debug)]
enum TargetTrait {
    Clone,
    ToOwned,
}

#[derive(Debug)]
struct CallCandidate<'tcx> {
    target: TargetTrait,
    kind: CallKind<'tcx>,
    // DefId of the called method from an impl block that implements the target trait
    method_def_id: DefId,
}

impl<'tcx> CallCandidate<'tcx> {
    #[inline]
    fn message(&self) -> &'static str {
        match self.target {
            TargetTrait::Clone => "assigning the result of `Clone::clone()` may be inefficient",
            TargetTrait::ToOwned => "assigning the result of `ToOwned::to_owned()` may be inefficient",
        }
    }

    #[inline]
    fn suggestion_msg(&self) -> &'static str {
        match self.target {
            TargetTrait::Clone => "use `clone_from()`",
            TargetTrait::ToOwned => "use `clone_into()`",
        }
    }

    fn suggested_replacement(
        &self,
        cx: &LateContext<'tcx>,
        lhs: &hir::Expr<'tcx>,
        applicability: &mut Applicability,
    ) -> String {
        match self.target {
            TargetTrait::Clone => {
                match self.kind {
                    CallKind::MethodCall { receiver } => {
                        let receiver_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                            // `*lhs = self_expr.clone();` -> `lhs.clone_from(self_expr)`
                            Sugg::hir_with_applicability(cx, ref_expr, "_", applicability)
                        } else {
                            // `lhs = self_expr.clone();` -> `lhs.clone_from(self_expr)`
                            Sugg::hir_with_applicability(cx, lhs, "_", applicability)
                        }
                        .maybe_par();

                        // Determine whether we need to reference the argument to clone_from().
                        let clone_receiver_type = cx.typeck_results().expr_ty(receiver);
                        let clone_receiver_adj_type = cx.typeck_results().expr_ty_adjusted(receiver);
                        let mut arg_sugg = Sugg::hir_with_applicability(cx, receiver, "_", applicability);
                        if clone_receiver_type != clone_receiver_adj_type {
                            // The receiver may have been a value type, so we need to add an `&` to
                            // be sure the argument to clone_from will be a reference.
                            arg_sugg = arg_sugg.addr();
                        };

                        format!("{receiver_sugg}.clone_from({arg_sugg})")
                    },
                    CallKind::FunctionCall { self_arg, .. } => {
                        let self_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                            // `*lhs = Clone::clone(self_expr);` -> `Clone::clone_from(lhs, self_expr)`
                            Sugg::hir_with_applicability(cx, ref_expr, "_", applicability)
                        } else {
                            // `lhs = Clone::clone(self_expr);` -> `Clone::clone_from(&mut lhs, self_expr)`
                            Sugg::hir_with_applicability(cx, lhs, "_", applicability).mut_addr()
                        };
                        // The RHS had to be exactly correct before the call, there is no auto-deref for function calls.
                        let rhs_sugg = Sugg::hir_with_applicability(cx, self_arg, "_", applicability);

                        format!("Clone::clone_from({self_sugg}, {rhs_sugg})")
                    },
                }
            },
            TargetTrait::ToOwned => {
                let rhs_sugg = if let ExprKind::Unary(hir::UnOp::Deref, ref_expr) = lhs.kind {
                    // `*lhs = rhs.to_owned()` -> `rhs.clone_into(lhs)`
                    // `*lhs = ToOwned::to_owned(rhs)` -> `ToOwned::clone_into(rhs, lhs)`
                    let sugg = Sugg::hir_with_applicability(cx, ref_expr, "_", applicability).maybe_par();
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
                    Sugg::hir_with_applicability(cx, lhs, "_", applicability)
                        .maybe_par()
                        .mut_addr()
                };

                match self.kind {
                    CallKind::MethodCall { receiver } => {
                        let receiver_sugg = Sugg::hir_with_applicability(cx, receiver, "_", applicability);
                        format!("{receiver_sugg}.clone_into({rhs_sugg})")
                    },
                    CallKind::FunctionCall { self_arg, .. } => {
                        let self_sugg = Sugg::hir_with_applicability(cx, self_arg, "_", applicability);
                        format!("ToOwned::clone_into({self_sugg}, {rhs_sugg})")
                    },
                }
            },
        }
    }
}
