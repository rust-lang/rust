use std::collections::VecDeque;

use clippy_utils::diagnostics::span_lint_and_sugg;
use itertools::{izip, Itertools};
use rustc_ast::{walk_list, Label, Mutability};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_hir::intravisit::{walk_expr, FnKind, Visitor};
use rustc_hir::{
    Arm, Block, Body, Expr, ExprKind, Guard, HirId, ImplicitSelfKind, Let, Local, Pat, PatKind, Path, PathSegment,
    QPath, Stmt, StmtKind, TyKind, UnOp,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_middle::ty::{Ty, TyCtxt, TypeckResults};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;
use rustc_span::symbol::Ident;
use rustc_span::Span;

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
    /// In some cases, this would not catch all useless arguments.
    ///
    /// ```rust
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
    /// ```rust
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
    /// ```rust
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
    #[clippy::version = "1.60.0"]
    pub ONLY_USED_IN_RECURSION,
    complexity,
    "arguments that is only used in recursion can be removed"
}
declare_lint_pass!(OnlyUsedInRecursion => [ONLY_USED_IN_RECURSION]);

impl<'tcx> LateLintPass<'tcx> for OnlyUsedInRecursion {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: FnKind<'tcx>,
        decl: &'tcx rustc_hir::FnDecl<'tcx>,
        body: &'tcx Body<'tcx>,
        _: Span,
        id: HirId,
    ) {
        if let FnKind::ItemFn(ident, ..) | FnKind::Method(ident, ..) = kind {
            let def_id = id.owner.to_def_id();
            let data = cx.tcx.def_path(def_id).data;

            if data.len() > 1 {
                match data.get(data.len() - 2) {
                    Some(DisambiguatedDefPathData {
                        data: DefPathData::Impl,
                        disambiguator,
                    }) if *disambiguator != 0 => return,
                    _ => {},
                }
            }

            let has_self = !matches!(decl.implicit_self, ImplicitSelfKind::None);

            let ty_res = cx.typeck_results();
            let param_span = body
                .params
                .iter()
                .flat_map(|param| {
                    let mut v = Vec::new();
                    param.pat.each_binding(|_, hir_id, span, ident| {
                        v.push((hir_id, span, ident));
                    });
                    v
                })
                .skip(if has_self { 1 } else { 0 })
                .filter(|(_, _, ident)| !ident.name.as_str().starts_with('_'))
                .collect_vec();

            let params = body.params.iter().map(|param| param.pat).collect();

            let mut visitor = SideEffectVisit {
                graph: FxHashMap::default(),
                has_side_effect: FxHashSet::default(),
                ret_vars: Vec::new(),
                contains_side_effect: false,
                break_vars: FxHashMap::default(),
                params,
                fn_ident: ident,
                fn_def_id: def_id,
                is_method: matches!(kind, FnKind::Method(..)),
                has_self,
                ty_res,
                ty_ctx: cx.tcx,
            };

            visitor.visit_expr(&body.value);
            let vars = std::mem::take(&mut visitor.ret_vars);
            // this would set the return variables to side effect
            visitor.add_side_effect(vars);

            let mut queue = visitor.has_side_effect.iter().copied().collect::<VecDeque<_>>();

            // a simple BFS to check all the variables that have side effect
            while let Some(id) = queue.pop_front() {
                if let Some(next) = visitor.graph.get(&id) {
                    for i in next {
                        if !visitor.has_side_effect.contains(i) {
                            visitor.has_side_effect.insert(*i);
                            queue.push_back(*i);
                        }
                    }
                }
            }

            for (id, span, ident) in param_span {
                // if the variable is not used in recursion, it would be marked as unused
                if !visitor.has_side_effect.contains(&id) {
                    let mut queue = VecDeque::new();
                    let mut visited = FxHashSet::default();

                    queue.push_back(id);

                    // a simple BFS to check the graph can reach to itself
                    // if it can't, it means the variable is never used in recursion
                    while let Some(id) = queue.pop_front() {
                        if let Some(next) = visitor.graph.get(&id) {
                            for i in next {
                                if !visited.contains(i) {
                                    visited.insert(id);
                                    queue.push_back(*i);
                                }
                            }
                        }
                    }

                    if visited.contains(&id) {
                        span_lint_and_sugg(
                            cx,
                            ONLY_USED_IN_RECURSION,
                            span,
                            "parameter is only used in recursion",
                            "if this is intentional, prefix with an underscore",
                            format!("_{}", ident.name.as_str()),
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
        }
    }
}

pub fn is_primitive(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str => true,
        ty::Ref(_, t, _) => is_primitive(*t),
        _ => false,
    }
}

pub fn is_array(ty: Ty<'_>) -> bool {
    match ty.kind() {
        ty::Array(..) | ty::Slice(..) => true,
        ty::Ref(_, t, _) => is_array(*t),
        _ => false,
    }
}

/// This builds the graph of side effect.
/// The edge `a -> b` means if `a` has side effect, `b` will have side effect.
///
/// There are some example in following code:
/// ```rust, ignore
/// let b = 1;
/// let a = b; // a -> b
/// let (c, d) = (a, b); // c -> b, d -> b
///
/// let e = if a == 0 { // e -> a
///     c // e -> c
/// } else {
///     d // e -> d
/// };
/// ```
pub struct SideEffectVisit<'tcx> {
    graph: FxHashMap<HirId, FxHashSet<HirId>>,
    has_side_effect: FxHashSet<HirId>,
    // bool for if the variable was dereferenced from mutable reference
    ret_vars: Vec<(HirId, bool)>,
    contains_side_effect: bool,
    // break label
    break_vars: FxHashMap<Ident, Vec<(HirId, bool)>>,
    params: Vec<&'tcx Pat<'tcx>>,
    fn_ident: Ident,
    fn_def_id: DefId,
    is_method: bool,
    has_self: bool,
    ty_res: &'tcx TypeckResults<'tcx>,
    ty_ctx: TyCtxt<'tcx>,
}

impl<'tcx> Visitor<'tcx> for SideEffectVisit<'tcx> {
    fn visit_block(&mut self, b: &'tcx Block<'tcx>) {
        b.stmts.iter().for_each(|stmt| {
            self.visit_stmt(stmt);
            self.ret_vars.clear();
        });
        walk_list!(self, visit_expr, b.expr);
    }

    fn visit_stmt(&mut self, s: &'tcx Stmt<'tcx>) {
        match s.kind {
            StmtKind::Local(Local {
                pat, init: Some(init), ..
            }) => {
                self.visit_pat_expr(pat, init, false);
                self.ret_vars.clear();
            },
            StmtKind::Item(i) => {
                let item = self.ty_ctx.hir().item(i);
                self.visit_item(item);
                self.ret_vars.clear();
            },
            StmtKind::Expr(e) | StmtKind::Semi(e) => {
                self.visit_expr(e);
                self.ret_vars.clear();
            },
            StmtKind::Local(_) => {},
        }
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        match ex.kind {
            ExprKind::Array(exprs) | ExprKind::Tup(exprs) => {
                self.ret_vars = exprs
                    .iter()
                    .flat_map(|expr| {
                        self.visit_expr(expr);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect();
            },
            ExprKind::Call(callee, args) => self.visit_fn(callee, args),
            ExprKind::MethodCall(path, args, _) => self.visit_method_call(path, args),
            ExprKind::Binary(_, lhs, rhs) => {
                self.visit_bin_op(lhs, rhs);
            },
            ExprKind::Unary(op, expr) => self.visit_un_op(op, expr),
            ExprKind::Let(Let { pat, init, .. }) => self.visit_pat_expr(pat, init, false),
            ExprKind::If(bind, then_expr, else_expr) => {
                self.visit_if(bind, then_expr, else_expr);
            },
            ExprKind::Match(expr, arms, _) => self.visit_match(expr, arms),
            // since analysing the closure is not easy, just set all variables in it to side-effect
            ExprKind::Closure(_, _, body_id, _, _) => {
                let body = self.ty_ctx.hir().body(body_id);
                self.visit_body(body);
                let vars = std::mem::take(&mut self.ret_vars);
                self.add_side_effect(vars);
            },
            ExprKind::Loop(block, label, _, _) | ExprKind::Block(block, label) => {
                self.visit_block_label(block, label);
            },
            ExprKind::Assign(bind, expr, _) => {
                self.visit_assign(bind, expr);
            },
            ExprKind::AssignOp(_, bind, expr) => {
                self.visit_assign(bind, expr);
                self.visit_bin_op(bind, expr);
            },
            ExprKind::Field(expr, _) => {
                self.visit_expr(expr);
                if matches!(self.ty_res.expr_ty(expr).kind(), ty::Ref(_, _, Mutability::Mut)) {
                    self.ret_vars.iter_mut().for_each(|(_, b)| *b = true);
                }
            },
            ExprKind::Index(expr, index) => {
                self.visit_expr(expr);
                let mut vars = std::mem::take(&mut self.ret_vars);
                self.visit_expr(index);
                self.ret_vars.append(&mut vars);

                if !is_array(self.ty_res.expr_ty(expr)) {
                    self.add_side_effect(self.ret_vars.clone());
                } else if matches!(self.ty_res.expr_ty(expr).kind(), ty::Ref(_, _, Mutability::Mut)) {
                    self.ret_vars.iter_mut().for_each(|(_, b)| *b = true);
                }
            },
            ExprKind::Break(dest, Some(expr)) => {
                self.visit_expr(expr);
                if let Some(label) = dest.label {
                    self.break_vars
                        .entry(label.ident)
                        .or_insert(Vec::new())
                        .append(&mut self.ret_vars);
                }
                self.contains_side_effect = true;
            },
            ExprKind::Ret(Some(expr)) => {
                self.visit_expr(expr);
                let vars = std::mem::take(&mut self.ret_vars);
                self.add_side_effect(vars);
                self.contains_side_effect = true;
            },
            ExprKind::Break(_, None) | ExprKind::Continue(_) | ExprKind::Ret(None) => {
                self.contains_side_effect = true;
            },
            ExprKind::Struct(_, exprs, expr) => {
                let mut ret_vars = exprs
                    .iter()
                    .flat_map(|field| {
                        self.visit_expr(field.expr);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect();

                walk_list!(self, visit_expr, expr);
                self.ret_vars.append(&mut ret_vars);
            },
            _ => walk_expr(self, ex),
        }
    }

    fn visit_path(&mut self, path: &'tcx Path<'tcx>, _id: HirId) {
        if let Res::Local(id) = path.res {
            self.ret_vars.push((id, false));
        }
    }
}

impl<'tcx> SideEffectVisit<'tcx> {
    fn visit_assign(&mut self, lhs: &'tcx Expr<'tcx>, rhs: &'tcx Expr<'tcx>) {
        // Just support array and tuple unwrapping for now.
        //
        // ex) `(a, b) = (c, d);`
        // The graph would look like this:
        //   a -> c
        //   b -> d
        //
        // This would minimize the connection of the side-effect graph.
        match (&lhs.kind, &rhs.kind) {
            (ExprKind::Array(lhs), ExprKind::Array(rhs)) | (ExprKind::Tup(lhs), ExprKind::Tup(rhs)) => {
                // if not, it is a compile error
                debug_assert!(lhs.len() == rhs.len());
                izip!(*lhs, *rhs).for_each(|(lhs, rhs)| self.visit_assign(lhs, rhs));
            },
            // in other assigns, we have to connect all each other
            // because they can be connected somehow
            _ => {
                self.visit_expr(lhs);
                let lhs_vars = std::mem::take(&mut self.ret_vars);
                self.visit_expr(rhs);
                let rhs_vars = std::mem::take(&mut self.ret_vars);
                self.connect_assign(&lhs_vars, &rhs_vars, false);
            },
        }
    }

    fn visit_block_label(&mut self, block: &'tcx Block<'tcx>, label: Option<Label>) {
        self.visit_block(block);
        let _ = label.and_then(|label| {
            self.break_vars
                .remove(&label.ident)
                .map(|mut break_vars| self.ret_vars.append(&mut break_vars))
        });
    }

    fn visit_bin_op(&mut self, lhs: &'tcx Expr<'tcx>, rhs: &'tcx Expr<'tcx>) {
        self.visit_expr(lhs);
        let mut ret_vars = std::mem::take(&mut self.ret_vars);
        self.visit_expr(rhs);
        self.ret_vars.append(&mut ret_vars);

        // the binary operation between non primitive values are overloaded operators
        // so they can have side-effects
        if !is_primitive(self.ty_res.expr_ty(lhs)) || !is_primitive(self.ty_res.expr_ty(rhs)) {
            self.ret_vars.iter().for_each(|id| {
                self.has_side_effect.insert(id.0);
            });
            self.contains_side_effect = true;
        }
    }

    fn visit_un_op(&mut self, op: UnOp, expr: &'tcx Expr<'tcx>) {
        self.visit_expr(expr);
        let ty = self.ty_res.expr_ty(expr);
        // dereferencing a reference has no side-effect
        if !is_primitive(ty) && !matches!((op, ty.kind()), (UnOp::Deref, ty::Ref(..))) {
            self.add_side_effect(self.ret_vars.clone());
        }

        if matches!((op, ty.kind()), (UnOp::Deref, ty::Ref(_, _, Mutability::Mut))) {
            self.ret_vars.iter_mut().for_each(|(_, b)| *b = true);
        }
    }

    fn visit_pat_expr(&mut self, pat: &'tcx Pat<'tcx>, expr: &'tcx Expr<'tcx>, connect_self: bool) {
        match (&pat.kind, &expr.kind) {
            (PatKind::Tuple(pats, _), ExprKind::Tup(exprs)) => {
                self.ret_vars = izip!(*pats, *exprs)
                    .flat_map(|(pat, expr)| {
                        self.visit_pat_expr(pat, expr, connect_self);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect();
            },
            (PatKind::Slice(front_exprs, _, back_exprs), ExprKind::Array(exprs)) => {
                let mut vars = izip!(*front_exprs, *exprs)
                    .flat_map(|(pat, expr)| {
                        self.visit_pat_expr(pat, expr, connect_self);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect();
                self.ret_vars = izip!(back_exprs.iter().rev(), exprs.iter().rev())
                    .flat_map(|(pat, expr)| {
                        self.visit_pat_expr(pat, expr, connect_self);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect();
                self.ret_vars.append(&mut vars);
            },
            _ => {
                let mut lhs_vars = Vec::new();
                pat.each_binding(|_, id, _, _| lhs_vars.push((id, false)));
                self.visit_expr(expr);
                let rhs_vars = std::mem::take(&mut self.ret_vars);
                self.connect_assign(&lhs_vars, &rhs_vars, connect_self);
                self.ret_vars = rhs_vars;
            },
        }
    }

    fn visit_fn(&mut self, callee: &'tcx Expr<'tcx>, args: &'tcx [Expr<'tcx>]) {
        self.visit_expr(callee);
        let mut ret_vars = std::mem::take(&mut self.ret_vars);
        self.add_side_effect(ret_vars.clone());

        let mut is_recursive = false;

        if_chain! {
            if !self.has_self;
            if let ExprKind::Path(QPath::Resolved(_, path)) = callee.kind;
            if let Res::Def(DefKind::Fn, def_id) = path.res;
            if self.fn_def_id == def_id;
            then {
                is_recursive = true;
            }
        }

        if_chain! {
            if !self.has_self && self.is_method;
            if let ExprKind::Path(QPath::TypeRelative(ty, segment)) = callee.kind;
            if segment.ident == self.fn_ident;
            if let TyKind::Path(QPath::Resolved(_, path)) = ty.kind;
            if let Res::SelfTy{ .. } = path.res;
            then {
                is_recursive = true;
            }
        }

        if is_recursive {
            izip!(self.params.clone(), args).for_each(|(pat, expr)| {
                self.visit_pat_expr(pat, expr, true);
                self.ret_vars.clear();
            });
        } else {
            // This would set arguments used in closure that does not have side-effect.
            // Closure itself can be detected whether there is a side-effect, but the
            // value of variable that is holding closure can change.
            // So, we just check the variables.
            self.ret_vars = args
                .iter()
                .flat_map(|expr| {
                    self.visit_expr(expr);
                    std::mem::take(&mut self.ret_vars)
                })
                .collect_vec()
                .into_iter()
                .map(|id| {
                    self.has_side_effect.insert(id.0);
                    id
                })
                .collect();
            self.contains_side_effect = true;
        }

        self.ret_vars.append(&mut ret_vars);
    }

    fn visit_method_call(&mut self, path: &'tcx PathSegment<'tcx>, args: &'tcx [Expr<'tcx>]) {
        if_chain! {
            if self.is_method;
            if path.ident == self.fn_ident;
            if let ExprKind::Path(QPath::Resolved(_, path)) = args.first().unwrap().kind;
            if let Res::Local(..) = path.res;
            let ident = path.segments.last().unwrap().ident;
            if ident.name == kw::SelfLower;
            then {
                izip!(self.params.clone(), args.iter())
                    .for_each(|(pat, expr)| {
                        self.visit_pat_expr(pat, expr, true);
                        self.ret_vars.clear();
                    });
            } else {
                self.ret_vars = args
                    .iter()
                    .flat_map(|expr| {
                        self.visit_expr(expr);
                        std::mem::take(&mut self.ret_vars)
                    })
                    .collect_vec()
                    .into_iter()
                    .map(|a| {
                        self.has_side_effect.insert(a.0);
                        a
                    })
                    .collect();
                self.contains_side_effect = true;
            }
        }
    }

    fn visit_if(&mut self, bind: &'tcx Expr<'tcx>, then_expr: &'tcx Expr<'tcx>, else_expr: Option<&'tcx Expr<'tcx>>) {
        let contains_side_effect = self.contains_side_effect;
        self.contains_side_effect = false;
        self.visit_expr(bind);
        let mut vars = std::mem::take(&mut self.ret_vars);
        self.visit_expr(then_expr);
        let mut then_vars = std::mem::take(&mut self.ret_vars);
        walk_list!(self, visit_expr, else_expr);
        if self.contains_side_effect {
            self.add_side_effect(vars.clone());
        }
        self.contains_side_effect |= contains_side_effect;
        self.ret_vars.append(&mut vars);
        self.ret_vars.append(&mut then_vars);
    }

    fn visit_match(&mut self, expr: &'tcx Expr<'tcx>, arms: &'tcx [Arm<'tcx>]) {
        self.visit_expr(expr);
        let mut expr_vars = std::mem::take(&mut self.ret_vars);
        self.ret_vars = arms
            .iter()
            .flat_map(|arm| {
                let contains_side_effect = self.contains_side_effect;
                self.contains_side_effect = false;
                // this would visit `expr` multiple times
                // but couldn't think of a better way
                self.visit_pat_expr(arm.pat, expr, false);
                let mut vars = std::mem::take(&mut self.ret_vars);
                let _ = arm.guard.as_ref().map(|guard| {
                    self.visit_expr(match guard {
                        Guard::If(expr) | Guard::IfLet(_, expr) => expr,
                    });
                    vars.append(&mut self.ret_vars);
                });
                self.visit_expr(arm.body);
                if self.contains_side_effect {
                    self.add_side_effect(vars.clone());
                    self.add_side_effect(expr_vars.clone());
                }
                self.contains_side_effect |= contains_side_effect;
                vars.append(&mut self.ret_vars);
                vars
            })
            .collect();
        self.ret_vars.append(&mut expr_vars);
    }

    fn connect_assign(&mut self, lhs: &[(HirId, bool)], rhs: &[(HirId, bool)], connect_self: bool) {
        // if mutable dereference is on assignment it can have side-effect
        // (this can lead to parameter mutable dereference and change the original value)
        // too hard to detect whether this value is from parameter, so this would all
        // check mutable dereference assignment to side effect
        lhs.iter().filter(|(_, b)| *b).for_each(|(id, _)| {
            self.has_side_effect.insert(*id);
            self.contains_side_effect = true;
        });

        // there is no connection
        if lhs.is_empty() || rhs.is_empty() {
            return;
        }

        // by connected rhs in cycle, the connections would decrease
        // from `n * m` to `n + m`
        // where `n` and `m` are length of `lhs` and `rhs`.

        // unwrap is possible since rhs is not empty
        let rhs_first = rhs.first().unwrap();
        for (id, _) in lhs.iter() {
            if connect_self || *id != rhs_first.0 {
                self.graph
                    .entry(*id)
                    .or_insert_with(FxHashSet::default)
                    .insert(rhs_first.0);
            }
        }

        let rhs = rhs.iter();
        izip!(rhs.clone().cycle().skip(1), rhs).for_each(|(from, to)| {
            if connect_self || from.0 != to.0 {
                self.graph.entry(from.0).or_insert_with(FxHashSet::default).insert(to.0);
            }
        });
    }

    fn add_side_effect(&mut self, v: Vec<(HirId, bool)>) {
        for (id, _) in v {
            self.has_side_effect.insert(id);
            self.contains_side_effect = true;
        }
    }
}
