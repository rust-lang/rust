use ControlFlow::{Break, Continue};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{fn_def_id, get_enclosing_block, match_any_def_paths, match_def_path, path_to_local_id, paths};
use rustc_ast::Mutability;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr, walk_local};
use rustc_hir::{Expr, ExprKind, HirId, LetStmt, Node, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_session::declare_lint_pass;
use rustc_span::sym;
use std::ops::ControlFlow;

declare_clippy_lint! {
    /// ### What it does
    /// Looks for code that spawns a process but never calls `wait()` on the child.
    ///
    /// ### Why is this bad?
    /// As explained in the [standard library documentation](https://doc.rust-lang.org/stable/std/process/struct.Child.html#warning),
    /// calling `wait()` is necessary on Unix platforms to properly release all OS resources associated with the process.
    /// Not doing so will effectively leak process IDs and/or other limited global resources,
    /// which can eventually lead to resource exhaustion, so it's recommended to call `wait()` in long-running applications.
    /// Such processes are called "zombie processes".
    ///
    /// ### Example
    /// ```rust
    /// use std::process::Command;
    ///
    /// let _child = Command::new("ls").spawn().expect("failed to execute child");
    /// ```
    /// Use instead:
    /// ```rust
    /// use std::process::Command;
    ///
    /// let mut child = Command::new("ls").spawn().expect("failed to execute child");
    /// child.wait().expect("failed to wait on child");
    /// ```
    #[clippy::version = "1.74.0"]
    pub ZOMBIE_PROCESSES,
    suspicious,
    "not waiting on a spawned child process"
}
declare_lint_pass!(ZombieProcesses => [ZOMBIE_PROCESSES]);

impl<'tcx> LateLintPass<'tcx> for ZombieProcesses {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(..) | ExprKind::MethodCall(..) = expr.kind
            && let Some(child_adt) = cx.typeck_results().expr_ty(expr).ty_adt_def()
            && match_def_path(cx, child_adt.did(), &paths::CHILD)
        {
            match cx.tcx.parent_hir_node(expr.hir_id) {
                Node::LetStmt(local)
                    if let PatKind::Binding(_, local_id, ..) = local.pat.kind
                        && let Some(enclosing_block) = get_enclosing_block(cx, expr.hir_id) =>
                {
                    let mut vis = WaitFinder::WalkUpTo(cx, local_id);

                    // If it does have a `wait()` call, we're done. Don't lint.
                    if let Break(BreakReason::WaitFound) = walk_block(&mut vis, enclosing_block) {
                        return;
                    }

                    // Don't emit a suggestion since the binding is used later
                    check(cx, expr, false);
                },
                Node::LetStmt(&LetStmt { pat, .. }) if let PatKind::Wild = pat.kind => {
                    // `let _ = child;`, also dropped immediately without `wait()`ing
                    check(cx, expr, true);
                },
                Node::Stmt(&Stmt {
                    kind: StmtKind::Semi(_),
                    ..
                }) => {
                    // Immediately dropped. E.g. `std::process::Command::new("echo").spawn().unwrap();`
                    check(cx, expr, true);
                },
                _ => {},
            }
        }
    }
}

enum BreakReason {
    WaitFound,
    EarlyReturn,
}

/// A visitor responsible for finding a `wait()` call on a local variable.
///
/// Conditional `wait()` calls are assumed to not call wait:
/// ```ignore
/// let mut c = Command::new("").spawn().unwrap();
/// if true {
///     c.wait();
/// }
/// ```
///
/// Note that this visitor does NOT explicitly look for `wait()` calls directly, but rather does the
/// inverse -- checking if all uses of the local are either:
/// - a field access (`child.{stderr,stdin,stdout}`)
/// - calling `id` or `kill`
/// - no use at all (e.g. `let _x = child;`)
/// - taking a shared reference (`&`), `wait()` can't go through that
///
/// None of these are sufficient to prevent zombie processes.
/// Doing it like this means more FNs, but FNs are better than FPs.
///
/// `return` expressions, conditional or not, short-circuit the visitor because
/// if a `wait()` call hadn't been found at that point, it might never reach one at a later point:
/// ```ignore
/// let mut c = Command::new("").spawn().unwrap();
/// if true {
///     return; // Break(BreakReason::EarlyReturn)
/// }
/// c.wait(); // this might not be reachable
/// ```
enum WaitFinder<'a, 'tcx> {
    WalkUpTo(&'a LateContext<'tcx>, HirId),
    Found(&'a LateContext<'tcx>, HirId),
}

impl<'tcx> Visitor<'tcx> for WaitFinder<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;
    type Result = ControlFlow<BreakReason>;

    fn visit_local(&mut self, l: &'tcx LetStmt<'tcx>) -> Self::Result {
        if let Self::WalkUpTo(cx, local_id) = *self
            && let PatKind::Binding(_, pat_id, ..) = l.pat.kind
            && local_id == pat_id
        {
            *self = Self::Found(cx, local_id);
        }

        walk_local(self, l)
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) -> Self::Result {
        let Self::Found(cx, local_id) = *self else {
            return walk_expr(self, ex);
        };

        if path_to_local_id(ex, local_id) {
            match cx.tcx.parent_hir_node(ex.hir_id) {
                Node::Stmt(Stmt {
                    kind: StmtKind::Semi(_),
                    ..
                }) => {},
                Node::Expr(expr) if let ExprKind::Field(..) = expr.kind => {},
                Node::Expr(expr) if let ExprKind::AddrOf(_, Mutability::Not, _) = expr.kind => {},
                Node::Expr(expr)
                    if let Some(fn_did) = fn_def_id(cx, expr)
                        && match_any_def_paths(cx, fn_did, &[&paths::CHILD_ID, &paths::CHILD_KILL]).is_some() => {},

                // Conservatively assume that all other kinds of nodes call `.wait()` somehow.
                _ => return Break(BreakReason::WaitFound),
            }
        } else {
            match ex.kind {
                ExprKind::Ret(..) => return Break(BreakReason::EarlyReturn),
                ExprKind::If(cond, then, None) => {
                    walk_expr(self, cond)?;

                    // A `wait()` call in an if expression with no else is not enough:
                    //
                    // let c = spawn();
                    // if true {
                    //   c.wait();
                    // }
                    //
                    // This might not call wait(). However, early returns are propagated,
                    // because they might lead to a later wait() not being called.
                    if let Break(BreakReason::EarlyReturn) = walk_expr(self, then) {
                        return Break(BreakReason::EarlyReturn);
                    }

                    return Continue(());
                },

                ExprKind::If(cond, then, Some(else_)) => {
                    walk_expr(self, cond)?;

                    #[expect(clippy::unnested_or_patterns)]
                    match (walk_expr(self, then), walk_expr(self, else_)) {
                        (Continue(()), Continue(()))

                        // `wait()` in one branch but nothing in the other does not count
                        | (Continue(()), Break(BreakReason::WaitFound))
                        | (Break(BreakReason::WaitFound), Continue(())) => {},

                        // `wait()` in both branches is ok
                        (Break(BreakReason::WaitFound), Break(BreakReason::WaitFound)) => {
                            return Break(BreakReason::WaitFound);
                        },

                        // Propagate early returns in either branch
                        (Break(BreakReason::EarlyReturn), _) | (_, Break(BreakReason::EarlyReturn)) => {
                            return Break(BreakReason::EarlyReturn);
                        },
                    }

                    return Continue(());
                },
                _ => {},
            }
        }

        walk_expr(self, ex)
    }

    fn nested_visit_map(&mut self) -> Self::Map {
        let (Self::Found(cx, _) | Self::WalkUpTo(cx, _)) = self;
        cx.tcx.hir()
    }
}

/// This function has shared logic between the different kinds of nodes that can trigger the lint.
///
/// In particular, `let <binding> = <expr that spawns child>;` requires some custom additional logic
/// such as checking that the binding is not used in certain ways, which isn't necessary for
/// `let _ = <expr that spawns child>;`.
///
/// This checks if the program doesn't unconditionally exit after the spawn expression.
fn check<'tcx>(cx: &LateContext<'tcx>, spawn_expr: &'tcx Expr<'tcx>, emit_suggestion: bool) {
    let Some(block) = get_enclosing_block(cx, spawn_expr.hir_id) else {
        return;
    };

    let mut vis = ExitPointFinder {
        cx,
        state: ExitPointState::WalkUpTo(spawn_expr.hir_id),
    };
    if let Break(ExitCallFound) = vis.visit_block(block) {
        // Visitor found an unconditional `exit()` call, so don't lint.
        return;
    }

    span_lint_and_then(
        cx,
        ZOMBIE_PROCESSES,
        spawn_expr.span,
        "spawned process is never `wait()`ed on",
        |diag| {
            if emit_suggestion {
                diag.span_suggestion(
                    spawn_expr.span.shrink_to_hi(),
                    "try",
                    ".wait()",
                    Applicability::MaybeIncorrect,
                );
            } else {
                diag.note("consider calling `.wait()`");
            }

            diag.note("not doing so might leave behind zombie processes")
                .note("see https://doc.rust-lang.org/stable/std/process/struct.Child.html#warning");
        },
    );
}

/// Checks if the given expression exits the process.
fn is_exit_expression(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    fn_def_id(cx, expr).is_some_and(|fn_did| {
        cx.tcx.is_diagnostic_item(sym::process_exit, fn_did) || match_def_path(cx, fn_did, &paths::ABORT)
    })
}

#[derive(Debug)]
enum ExitPointState {
    /// Still walking up to the expression that initiated the visitor.
    WalkUpTo(HirId),
    /// We're inside of a control flow construct (e.g. `if`, `match`, `loop`)
    /// Within this, we shouldn't accept any `exit()` calls in here, but we can leave all of these
    /// constructs later and still continue looking for an `exit()` call afterwards. Example:
    /// ```ignore
    /// Command::new("").spawn().unwrap();
    ///
    /// if true {                // depth=1
    ///     if true {            // depth=2
    ///         match () {       // depth=3
    ///             () => loop { // depth=4
    ///
    ///                 std::process::exit();
    ///                 ^^^^^^^^^^^^^^^^^^^^^ conditional exit call, ignored
    ///
    ///             }           // depth=3
    ///         }               // depth=2
    ///     }                   // depth=1
    /// }                       // depth=0
    ///
    /// std::process::exit();
    /// ^^^^^^^^^^^^^^^^^^^^^ this exit call is accepted because we're now unconditionally calling it
    /// ```
    /// We can only get into this state from `NoExit`.
    InControlFlow { depth: u32 },
    /// No exit call found yet, but looking for one.
    NoExit,
}

fn expr_enters_control_flow(expr: &Expr<'_>) -> bool {
    matches!(expr.kind, ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Loop(..))
}

struct ExitPointFinder<'a, 'tcx> {
    state: ExitPointState,
    cx: &'a LateContext<'tcx>,
}

struct ExitCallFound;

impl<'tcx> Visitor<'tcx> for ExitPointFinder<'_, 'tcx> {
    type Result = ControlFlow<ExitCallFound>;

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) -> Self::Result {
        match self.state {
            ExitPointState::WalkUpTo(id) if expr.hir_id == id => {
                self.state = ExitPointState::NoExit;
                walk_expr(self, expr)
            },
            ExitPointState::NoExit if expr_enters_control_flow(expr) => {
                self.state = ExitPointState::InControlFlow { depth: 1 };
                walk_expr(self, expr)?;
                if let ExitPointState::InControlFlow { .. } = self.state {
                    self.state = ExitPointState::NoExit;
                }
                Continue(())
            },
            ExitPointState::NoExit if is_exit_expression(self.cx, expr) => Break(ExitCallFound),
            ExitPointState::InControlFlow { ref mut depth } if expr_enters_control_flow(expr) => {
                *depth += 1;
                walk_expr(self, expr)?;
                match self.state {
                    ExitPointState::InControlFlow { depth: 1 } => self.state = ExitPointState::NoExit,
                    ExitPointState::InControlFlow { ref mut depth } => *depth -= 1,
                    _ => {},
                }
                Continue(())
            },
            _ => Continue(()),
        }
    }
}
