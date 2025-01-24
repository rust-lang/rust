use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{get_expr_use_or_unification_node, path_def_id, path_to_local, path_to_local_id};
use core::cell::Cell;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::HirIdMap;
use rustc_hir::{Body, Expr, ExprKind, HirId, ImplItem, ImplItemKind, Node, PatKind, TraitItem, TraitItemKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, ConstKind, EarlyBinder, GenericArgKind, GenericArgsRef};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::{Ident, kw};
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for arguments that are only used in recursion with no side-effects.
    ///
    /// ### Why is this bad?
    /// It could contain a useless calculation and can make function simpler.
    ///
    /// The arguments can be involved in calculations and assignments but as long as
    /// the calculations have no side-effects (function calls or mutating dereference)
    /// and the assigned variables are also only in recursion, it is useless.
    ///
    /// ### Known problems
    /// Too many code paths in the linting code are currently untested and prone to produce false
    /// positives or are prone to have performance implications.
    ///
    /// In some cases, this would not catch all useless arguments.
    ///
    /// ```no_run
    /// fn foo(a: usize, b: usize) -> usize {
    ///     let f = |x| x + 1;
    ///
    ///     if a == 0 {
    ///         1
    ///     } else {
    ///         foo(a - 1, f(b))
    ///     }
    /// }
    /// ```
    ///
    /// For example, the argument `b` is only used in recursion, but the lint would not catch it.
    ///
    /// List of some examples that can not be caught:
    /// - binary operation of non-primitive types
    /// - closure usage
    /// - some `break` relative operations
    /// - struct pattern binding
    ///
    /// Also, when you recurse the function name with path segments, it is not possible to detect.
    ///
    /// ### Example
    /// ```no_run
    /// fn f(a: usize, b: usize) -> usize {
    ///     if a == 0 {
    ///         1
    ///     } else {
    ///         f(a - 1, b + 1)
    ///     }
    /// }
    /// # fn main() {
    /// #     print!("{}", f(1, 1));
    /// # }
    /// ```
    /// Use instead:
    /// ```no_run
    /// fn f(a: usize) -> usize {
    ///     if a == 0 {
    ///         1
    ///     } else {
    ///         f(a - 1)
    ///     }
    /// }
    /// # fn main() {
    /// #     print!("{}", f(1));
    /// # }
    /// ```
    #[clippy::version = "1.61.0"]
    pub ONLY_USED_IN_RECURSION,
    complexity,
    "arguments that is only used in recursion can be removed"
}
impl_lint_pass!(OnlyUsedInRecursion => [ONLY_USED_IN_RECURSION]);

#[derive(Clone, Copy)]
enum FnKind {
    Fn,
    TraitFn,
    // This is a hack. Ideally we would store a `GenericArgsRef<'tcx>` type here, but a lint pass must be `'static`.
    // Substitutions are, however, interned. This allows us to store the pointer as a `usize` when comparing for
    // equality.
    ImplTraitFn(usize),
}

struct Param {
    /// The function this is a parameter for.
    fn_id: DefId,
    fn_kind: FnKind,
    /// The index of this parameter.
    idx: usize,
    ident: Ident,
    /// Whether this parameter should be linted. Set by `Params::flag_for_linting`.
    apply_lint: Cell<bool>,
    /// All the uses of this parameter.
    uses: Vec<Usage>,
}
impl Param {
    fn new(fn_id: DefId, fn_kind: FnKind, idx: usize, ident: Ident) -> Self {
        Self {
            fn_id,
            fn_kind,
            idx,
            ident,
            apply_lint: Cell::new(true),
            uses: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Usage {
    span: Span,
    idx: usize,
}
impl Usage {
    fn new(span: Span, idx: usize) -> Self {
        Self { span, idx }
    }
}

/// The parameters being checked by the lint, indexed by both the parameter's `HirId` and the
/// `DefId` of the function paired with the parameter's index.
#[derive(Default)]
#[allow(clippy::struct_field_names)]
struct Params {
    params: Vec<Param>,
    by_id: HirIdMap<usize>,
    by_fn: FxHashMap<(DefId, usize), usize>,
}
impl Params {
    fn insert(&mut self, param: Param, id: HirId) {
        let idx = self.params.len();
        self.by_id.insert(id, idx);
        self.by_fn.insert((param.fn_id, param.idx), idx);
        self.params.push(param);
    }

    fn remove_by_id(&mut self, id: HirId) {
        if let Some(param) = self.get_by_id_mut(id) {
            param.uses = Vec::new();
            let key = (param.fn_id, param.idx);
            self.by_fn.remove(&key);
            // FIXME(rust/#120456) - is `swap_remove` correct?
            self.by_id.swap_remove(&id);
        }
    }

    fn get_by_id_mut(&mut self, id: HirId) -> Option<&mut Param> {
        self.params.get_mut(*self.by_id.get(&id)?)
    }

    fn get_by_fn(&self, id: DefId, idx: usize) -> Option<&Param> {
        self.params.get(*self.by_fn.get(&(id, idx))?)
    }

    fn clear(&mut self) {
        self.params.clear();
        self.by_id.clear();
        self.by_fn.clear();
    }

    /// Sets the `apply_lint` flag on each parameter.
    fn flag_for_linting(&mut self) {
        // Stores the list of parameters currently being resolved. Needed to avoid cycles.
        let mut eval_stack = Vec::new();
        for param in &self.params {
            self.try_disable_lint_for_param(param, &mut eval_stack);
        }
    }

    // Use by calling `flag_for_linting`.
    fn try_disable_lint_for_param(&self, param: &Param, eval_stack: &mut Vec<usize>) -> bool {
        if !param.apply_lint.get() {
            true
        } else if param.uses.is_empty() {
            // Don't lint on unused parameters.
            param.apply_lint.set(false);
            true
        } else if eval_stack.contains(&param.idx) {
            // Already on the evaluation stack. Returning false will continue to evaluate other dependencies.
            false
        } else {
            eval_stack.push(param.idx);
            // Check all cases when used at a different parameter index.
            // Needed to catch cases like: `fn f(x: u32, y: u32) { f(y, x) }`
            for usage in param.uses.iter().filter(|u| u.idx != param.idx) {
                if self
                    .get_by_fn(param.fn_id, usage.idx)
                    // If the parameter can't be found, then it's used for more than just recursion.
                    .is_none_or(|p| self.try_disable_lint_for_param(p, eval_stack))
                {
                    param.apply_lint.set(false);
                    eval_stack.pop();
                    return true;
                }
            }
            eval_stack.pop();
            false
        }
    }
}

#[derive(Default)]
pub struct OnlyUsedInRecursion {
    /// Track the top-level body entered. Needed to delay reporting when entering nested bodies.
    entered_body: Option<HirId>,
    params: Params,
}

impl<'tcx> LateLintPass<'tcx> for OnlyUsedInRecursion {
    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        if body.value.span.from_expansion() {
            return;
        }
        // `skip_params` is either `0` or `1` to skip the `self` parameter in trait functions.
        // It can't be renamed, and it can't be removed without removing it from multiple functions.
        let (fn_id, fn_kind, skip_params) = match cx.tcx.parent_hir_node(body.value.hir_id) {
            Node::Item(i) => (i.owner_id.to_def_id(), FnKind::Fn, 0),
            Node::TraitItem(&TraitItem {
                kind: TraitItemKind::Fn(ref sig, _),
                owner_id,
                ..
            }) => (
                owner_id.to_def_id(),
                FnKind::TraitFn,
                usize::from(sig.decl.implicit_self.has_implicit_self()),
            ),
            Node::ImplItem(&ImplItem {
                kind: ImplItemKind::Fn(ref sig, _),
                owner_id,
                ..
            }) => {
                if let Node::Item(item) = cx.tcx.parent_hir_node(owner_id.into())
                    && let Some(trait_ref) = cx
                        .tcx
                        .impl_trait_ref(item.owner_id)
                        .map(EarlyBinder::instantiate_identity)
                    && let Some(trait_item_id) = cx.tcx.associated_item(owner_id).trait_item_def_id
                {
                    (
                        trait_item_id,
                        FnKind::ImplTraitFn(std::ptr::from_ref(cx.tcx.erase_regions(trait_ref.args)) as usize),
                        usize::from(sig.decl.implicit_self.has_implicit_self()),
                    )
                } else {
                    (owner_id.to_def_id(), FnKind::Fn, 0)
                }
            },
            _ => return,
        };
        body.params
            .iter()
            .enumerate()
            .skip(skip_params)
            .filter_map(|(idx, p)| match p.pat.kind {
                PatKind::Binding(_, id, ident, None) if !ident.as_str().starts_with('_') => {
                    Some((id, Param::new(fn_id, fn_kind, idx, ident)))
                },
                _ => None,
            })
            .for_each(|(id, param)| self.params.insert(param, id));
        if self.entered_body.is_none() {
            self.entered_body = Some(body.value.hir_id);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'tcx>) {
        if let Some(id) = path_to_local(e)
            && let Some(param) = self.params.get_by_id_mut(id)
        {
            let typeck = cx.typeck_results();
            let span = e.span;
            let mut e = e;
            loop {
                match get_expr_use_or_unification_node(cx.tcx, e) {
                    None | Some((Node::Stmt(_), _)) => return,
                    Some((Node::Expr(parent), child_id)) => match parent.kind {
                        // Recursive call. Track which index the parameter is used in.
                        ExprKind::Call(callee, args)
                            if path_def_id(cx, callee).is_some_and(|id| {
                                id == param.fn_id && has_matching_args(param.fn_kind, typeck.node_args(callee.hir_id))
                            }) =>
                        {
                            if let Some(idx) = args.iter().position(|arg| arg.hir_id == child_id) {
                                param.uses.push(Usage::new(span, idx));
                            }
                            return;
                        },
                        ExprKind::MethodCall(_, receiver, args, _)
                            if typeck.type_dependent_def_id(parent.hir_id).is_some_and(|id| {
                                id == param.fn_id && has_matching_args(param.fn_kind, typeck.node_args(parent.hir_id))
                            }) =>
                        {
                            if let Some(idx) = iter::once(receiver).chain(args).position(|arg| arg.hir_id == child_id) {
                                param.uses.push(Usage::new(span, idx));
                            }
                            return;
                        },
                        // Assignment to a parameter is fine.
                        ExprKind::Assign(lhs, _, _) | ExprKind::AssignOp(_, lhs, _) if lhs.hir_id == child_id => {
                            return;
                        },
                        // Parameter update e.g. `x = x + 1`
                        ExprKind::Assign(lhs, rhs, _) | ExprKind::AssignOp(_, lhs, rhs)
                            if rhs.hir_id == child_id && path_to_local_id(lhs, id) =>
                        {
                            return;
                        },
                        // Side-effect free expressions. Walk to the parent expression.
                        ExprKind::Binary(_, lhs, rhs)
                            if typeck.expr_ty(lhs).is_primitive() && typeck.expr_ty(rhs).is_primitive() =>
                        {
                            e = parent;
                            continue;
                        },
                        ExprKind::Unary(_, arg) if typeck.expr_ty(arg).is_primitive() => {
                            e = parent;
                            continue;
                        },
                        ExprKind::AddrOf(..) | ExprKind::Cast(..) => {
                            e = parent;
                            continue;
                        },
                        // Only allow field accesses without auto-deref
                        ExprKind::Field(..) if typeck.adjustments().get(child_id).is_none() => {
                            e = parent;
                            continue;
                        },
                        _ => (),
                    },
                    _ => (),
                }
                self.params.remove_by_id(id);
                return;
            }
        }
    }

    fn check_body_post(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        if self.entered_body == Some(body.value.hir_id) {
            self.entered_body = None;
            self.params.flag_for_linting();
            for param in &self.params.params {
                if param.apply_lint.get() {
                    span_lint_and_then(
                        cx,
                        ONLY_USED_IN_RECURSION,
                        param.ident.span,
                        "parameter is only used in recursion",
                        |diag| {
                            if param.ident.name != kw::SelfLower {
                                diag.span_suggestion(
                                    param.ident.span,
                                    "if this is intentional, prefix it with an underscore",
                                    format!("_{}", param.ident.name),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            diag.span_note(
                                param.uses.iter().map(|x| x.span).collect::<Vec<_>>(),
                                "parameter used here",
                            );
                        },
                    );
                }
            }
            self.params.clear();
        }
    }
}

fn has_matching_args(kind: FnKind, args: GenericArgsRef<'_>) -> bool {
    match kind {
        FnKind::Fn => true,
        FnKind::TraitFn => args.iter().enumerate().all(|(idx, subst)| match subst.unpack() {
            GenericArgKind::Lifetime(_) => true,
            GenericArgKind::Type(ty) => matches!(*ty.kind(), ty::Param(ty) if ty.index as usize == idx),
            GenericArgKind::Const(c) => matches!(c.kind(), ConstKind::Param(c) if c.index as usize == idx),
        }),
        FnKind::ImplTraitFn(expected_args) => std::ptr::from_ref(args) as usize == expected_args,
    }
}
