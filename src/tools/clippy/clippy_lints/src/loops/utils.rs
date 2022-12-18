use clippy_utils::ty::{has_iter_method, implements_trait};
use clippy_utils::{get_parent_expr, is_integer_const, path_to_local, path_to_local_id, sugg};
use if_chain::if_chain;
use rustc_ast::ast::{LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, walk_local, walk_pat, walk_stmt, Visitor};
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, HirId, HirIdMap, Local, Mutability, Pat, PatKind, Stmt};
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Ty};
use rustc_span::source_map::Spanned;
use rustc_span::symbol::{sym, Symbol};
use std::iter::Iterator;

#[derive(Debug, PartialEq, Eq)]
enum IncrementVisitorVarState {
    Initial,  // Not examined yet
    IncrOnce, // Incremented exactly once, may be a loop counter
    DontWarn,
}

/// Scan a for loop for variables that are incremented exactly once and not used after that.
pub(super) struct IncrementVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,                  // context reference
    states: HirIdMap<IncrementVisitorVarState>, // incremented variables
    depth: u32,                                 // depth of conditional expressions
}

impl<'a, 'tcx> IncrementVisitor<'a, 'tcx> {
    pub(super) fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            states: HirIdMap::default(),
            depth: 0,
        }
    }

    pub(super) fn into_results(self) -> impl Iterator<Item = HirId> {
        self.states.into_iter().filter_map(|(id, state)| {
            if state == IncrementVisitorVarState::IncrOnce {
                Some(id)
            } else {
                None
            }
        })
    }
}

impl<'a, 'tcx> Visitor<'tcx> for IncrementVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        // If node is a variable
        if let Some(def_id) = path_to_local(expr) {
            if let Some(parent) = get_parent_expr(self.cx, expr) {
                let state = self.states.entry(def_id).or_insert(IncrementVisitorVarState::Initial);
                if *state == IncrementVisitorVarState::IncrOnce {
                    *state = IncrementVisitorVarState::DontWarn;
                    return;
                }

                match parent.kind {
                    ExprKind::AssignOp(op, lhs, rhs) => {
                        if lhs.hir_id == expr.hir_id {
                            *state = if op.node == BinOpKind::Add
                                && is_integer_const(self.cx, rhs, 1)
                                && *state == IncrementVisitorVarState::Initial
                                && self.depth == 0
                            {
                                IncrementVisitorVarState::IncrOnce
                            } else {
                                // Assigned some other value or assigned multiple times
                                IncrementVisitorVarState::DontWarn
                            };
                        }
                    },
                    ExprKind::Assign(lhs, _, _) if lhs.hir_id == expr.hir_id => {
                        *state = IncrementVisitorVarState::DontWarn;
                    },
                    ExprKind::AddrOf(BorrowKind::Ref, mutability, _) if mutability == Mutability::Mut => {
                        *state = IncrementVisitorVarState::DontWarn;
                    },
                    _ => (),
                }
            }

            walk_expr(self, expr);
        } else if is_loop(expr) || is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
        } else if let ExprKind::Continue(_) = expr.kind {
            // If we see a `continue` block, then we increment depth so that the IncrementVisitor
            // state will be set to DontWarn if we see the variable being modified anywhere afterwards.
            self.depth += 1;
        } else {
            walk_expr(self, expr);
        }
    }
}

enum InitializeVisitorState<'hir> {
    Initial,                            // Not examined yet
    Declared(Symbol, Option<Ty<'hir>>), // Declared but not (yet) initialized
    Initialized {
        name: Symbol,
        ty: Option<Ty<'hir>>,
        initializer: &'hir Expr<'hir>,
    },
    DontWarn,
}

/// Checks whether a variable is initialized at the start of a loop and not modified
/// and used after the loop.
pub(super) struct InitializeVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,  // context reference
    end_expr: &'tcx Expr<'tcx>, // the for loop. Stop scanning here.
    var_id: HirId,
    state: InitializeVisitorState<'tcx>,
    depth: u32, // depth of conditional expressions
    past_loop: bool,
}

impl<'a, 'tcx> InitializeVisitor<'a, 'tcx> {
    pub(super) fn new(cx: &'a LateContext<'tcx>, end_expr: &'tcx Expr<'tcx>, var_id: HirId) -> Self {
        Self {
            cx,
            end_expr,
            var_id,
            state: InitializeVisitorState::Initial,
            depth: 0,
            past_loop: false,
        }
    }

    pub(super) fn get_result(&self) -> Option<(Symbol, Option<Ty<'tcx>>, &'tcx Expr<'tcx>)> {
        if let InitializeVisitorState::Initialized { name, ty, initializer } = self.state {
            Some((name, ty, initializer))
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for InitializeVisitor<'a, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn visit_local(&mut self, l: &'tcx Local<'_>) {
        // Look for declarations of the variable
        if_chain! {
            if l.pat.hir_id == self.var_id;
            if let PatKind::Binding(.., ident, _) = l.pat.kind;
            then {
                let ty = l.ty.map(|ty| hir_ty_to_ty(self.cx.tcx, ty));

                self.state = l.init.map_or(InitializeVisitorState::Declared(ident.name, ty), |init| {
                    InitializeVisitorState::Initialized {
                        initializer: init,
                        ty,
                        name: ident.name,
                    }
                })
            }
        }

        walk_local(self, l);
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if matches!(self.state, InitializeVisitorState::DontWarn) {
            return;
        }
        if expr.hir_id == self.end_expr.hir_id {
            self.past_loop = true;
            return;
        }
        // No need to visit expressions before the variable is
        // declared
        if matches!(self.state, InitializeVisitorState::Initial) {
            return;
        }

        // If node is the desired variable, see how it's used
        if path_to_local_id(expr, self.var_id) {
            if self.past_loop {
                self.state = InitializeVisitorState::DontWarn;
                return;
            }

            if let Some(parent) = get_parent_expr(self.cx, expr) {
                match parent.kind {
                    ExprKind::AssignOp(_, lhs, _) if lhs.hir_id == expr.hir_id => {
                        self.state = InitializeVisitorState::DontWarn;
                    },
                    ExprKind::Assign(lhs, rhs, _) if lhs.hir_id == expr.hir_id => {
                        self.state = if self.depth == 0 {
                            match self.state {
                                InitializeVisitorState::Declared(name, mut ty) => {
                                    if ty.is_none() {
                                        if let ExprKind::Lit(Spanned {
                                            node: LitKind::Int(_, LitIntType::Unsuffixed),
                                            ..
                                        }) = rhs.kind
                                        {
                                            ty = None;
                                        } else {
                                            ty = self.cx.typeck_results().expr_ty_opt(rhs);
                                        }
                                    }

                                    InitializeVisitorState::Initialized {
                                        initializer: rhs,
                                        ty,
                                        name,
                                    }
                                },
                                InitializeVisitorState::Initialized { ty, name, .. } => {
                                    InitializeVisitorState::Initialized {
                                        initializer: rhs,
                                        ty,
                                        name,
                                    }
                                },
                                _ => InitializeVisitorState::DontWarn,
                            }
                        } else {
                            InitializeVisitorState::DontWarn
                        }
                    },
                    ExprKind::AddrOf(BorrowKind::Ref, mutability, _) if mutability == Mutability::Mut => {
                        self.state = InitializeVisitorState::DontWarn;
                    },
                    _ => (),
                }
            }

            walk_expr(self, expr);
        } else if !self.past_loop && is_loop(expr) {
            self.state = InitializeVisitorState::DontWarn;
        } else if is_conditional(expr) {
            self.depth += 1;
            walk_expr(self, expr);
            self.depth -= 1;
        } else {
            walk_expr(self, expr);
        }
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        self.cx.tcx.hir()
    }
}

fn is_loop(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::Loop(..))
}

fn is_conditional(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::If(..) | ExprKind::Match(..))
}

#[derive(PartialEq, Eq)]
pub(super) enum Nesting {
    Unknown,     // no nesting detected yet
    RuledOut,    // the iterator is initialized or assigned within scope
    LookFurther, // no nesting detected, no further walk required
}

use self::Nesting::{LookFurther, RuledOut, Unknown};

pub(super) struct LoopNestVisitor {
    pub(super) hir_id: HirId,
    pub(super) iterator: HirId,
    pub(super) nesting: Nesting,
}

impl<'tcx> Visitor<'tcx> for LoopNestVisitor {
    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        if stmt.hir_id == self.hir_id {
            self.nesting = LookFurther;
        } else if self.nesting == Unknown {
            walk_stmt(self, stmt);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.nesting != Unknown {
            return;
        }
        if expr.hir_id == self.hir_id {
            self.nesting = LookFurther;
            return;
        }
        match expr.kind {
            ExprKind::Assign(path, _, _) | ExprKind::AssignOp(_, path, _) => {
                if path_to_local_id(path, self.iterator) {
                    self.nesting = RuledOut;
                }
            },
            _ => walk_expr(self, expr),
        }
    }

    fn visit_pat(&mut self, pat: &'tcx Pat<'_>) {
        if self.nesting != Unknown {
            return;
        }
        if let PatKind::Binding(_, id, ..) = pat.kind {
            if id == self.iterator {
                self.nesting = RuledOut;
                return;
            }
        }
        walk_pat(self, pat);
    }
}

/// If `arg` was the argument to a `for` loop, return the "cleanest" way of writing the
/// actual `Iterator` that the loop uses.
pub(super) fn make_iterator_snippet(cx: &LateContext<'_>, arg: &Expr<'_>, applic_ref: &mut Applicability) -> String {
    let impls_iterator = cx.tcx.get_diagnostic_item(sym::Iterator).map_or(false, |id| {
        implements_trait(cx, cx.typeck_results().expr_ty(arg), id, &[])
    });
    if impls_iterator {
        format!(
            "{}",
            sugg::Sugg::hir_with_applicability(cx, arg, "_", applic_ref).maybe_par()
        )
    } else {
        // (&x).into_iter() ==> x.iter()
        // (&mut x).into_iter() ==> x.iter_mut()
        let arg_ty = cx.typeck_results().expr_ty_adjusted(arg);
        match &arg_ty.kind() {
            ty::Ref(_, inner_ty, mutbl) if has_iter_method(cx, *inner_ty).is_some() => {
                let method_name = match mutbl {
                    Mutability::Mut => "iter_mut",
                    Mutability::Not => "iter",
                };
                let caller = match &arg.kind {
                    ExprKind::AddrOf(BorrowKind::Ref, _, arg_inner) => arg_inner,
                    _ => arg,
                };
                format!(
                    "{}.{method_name}()",
                    sugg::Sugg::hir_with_applicability(cx, caller, "_", applic_ref).maybe_par(),
                )
            },
            _ => format!(
                "{}.into_iter()",
                sugg::Sugg::hir_with_applicability(cx, arg, "_", applic_ref).maybe_par()
            ),
        }
    }
}
