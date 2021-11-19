use clippy_utils::ty::{has_iter_method, implements_trait};
use clippy_utils::{get_parent_expr, is_integer_const, path_to_local, path_to_local_id, sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_expr, walk_pat, walk_stmt, NestedVisitorMap, Visitor};
use rustc_hir::HirIdMap;
use rustc_hir::{BinOpKind, BorrowKind, Expr, ExprKind, HirId, Mutability, Pat, PatKind, Stmt, StmtKind};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::source_map::Span;
use rustc_span::symbol::{sym, Symbol};
use std::iter::Iterator;

#[derive(Debug, PartialEq)]
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
    done: bool,
}

impl<'a, 'tcx> IncrementVisitor<'a, 'tcx> {
    pub(super) fn new(cx: &'a LateContext<'tcx>) -> Self {
        Self {
            cx,
            states: HirIdMap::default(),
            depth: 0,
            done: false,
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
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        if self.done {
            return;
        }

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
            self.done = true;
        } else {
            walk_expr(self, expr);
        }
    }
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

enum InitializeVisitorState<'hir> {
    Initial,          // Not examined yet
    Declared(Symbol), // Declared but not (yet) initialized
    Initialized {
        name: Symbol,
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

    pub(super) fn get_result(&self) -> Option<(Symbol, &'tcx Expr<'tcx>)> {
        if let InitializeVisitorState::Initialized { name, initializer } = self.state {
            Some((name, initializer))
        } else {
            None
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for InitializeVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        // Look for declarations of the variable
        if_chain! {
            if let StmtKind::Local(local) = stmt.kind;
            if local.pat.hir_id == self.var_id;
            if let PatKind::Binding(.., ident, _) = local.pat.kind;
            then {
                self.state = local.init.map_or(InitializeVisitorState::Declared(ident.name), |init| {
                    InitializeVisitorState::Initialized {
                        initializer: init,
                        name: ident.name,
                    }
                })
            }
        }
        walk_stmt(self, stmt);
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
                        self.state = if_chain! {
                            if self.depth == 0;
                            if let InitializeVisitorState::Declared(name)
                                | InitializeVisitorState::Initialized { name, ..} = self.state;
                            then {
                                InitializeVisitorState::Initialized { initializer: rhs, name }
                            } else {
                                InitializeVisitorState::DontWarn
                            }
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

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
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
    type Map = Map<'tcx>;

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

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

// this function assumes the given expression is a `for` loop.
pub(super) fn get_span_of_entire_for_loop(expr: &Expr<'_>) -> Span {
    // for some reason this is the only way to get the `Span`
    // of the entire `for` loop
    if let ExprKind::Match(_, arms, _) = &expr.kind {
        arms[0].body.span
    } else {
        unreachable!()
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
        match &arg.kind {
            ExprKind::AddrOf(BorrowKind::Ref, mutability, arg_inner)
                if has_iter_method(cx, cx.typeck_results().expr_ty(arg_inner)).is_some() =>
            {
                let meth_name = match mutability {
                    Mutability::Mut => "iter_mut",
                    Mutability::Not => "iter",
                };
                format!(
                    "{}.{}()",
                    sugg::Sugg::hir_with_applicability(cx, arg_inner, "_", applic_ref).maybe_par(),
                    meth_name,
                )
            },
            _ => format!(
                "{}.into_iter()",
                sugg::Sugg::hir_with_applicability(cx, arg, "_", applic_ref).maybe_par()
            ),
        }
    }
}
