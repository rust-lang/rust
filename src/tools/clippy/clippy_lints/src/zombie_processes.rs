use ControlFlow::{Break, Continue};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::{fn_def_id, get_enclosing_block, path_to_local_id};
use rustc_ast::Mutability;
use rustc_ast::visit::visit_opt;
use rustc_errors::Applicability;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{Visitor, walk_block, walk_expr};
use rustc_hir::{Expr, ExprKind, HirId, LetStmt, Node, PatKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_session::declare_lint_pass;
use rustc_span::{Span, sym};
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
    /// To reduce the rate of false positives, if the spawned process is assigned to a binding, the lint actually works the other way around; it
    /// conservatively checks that all uses of a variable definitely don't call `wait()` and only then emits a warning.
    /// For that reason, a seemingly unrelated use can get called out as calling `wait()` in help messages.
    ///
    /// ### Control flow
    /// If a `wait()` call exists in an if/then block but not in the else block (or there is no else block),
    /// then this still gets linted as not calling `wait()` in all code paths.
    /// Likewise, when early-returning from the function, `wait()` calls that appear after the return expression
    /// are also not accepted.
    /// In other words, the `wait()` call must be unconditionally reachable after the spawn expression.
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
    #[clippy::version = "1.83.0"]
    pub ZOMBIE_PROCESSES,
    suspicious,
    "not waiting on a spawned child process"
}
declare_lint_pass!(ZombieProcesses => [ZOMBIE_PROCESSES]);

impl<'tcx> LateLintPass<'tcx> for ZombieProcesses {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        if let ExprKind::Call(..) | ExprKind::MethodCall(..) = expr.kind
            && let Some(child_adt) = cx.typeck_results().expr_ty(expr).ty_adt_def()
            && cx.tcx.is_diagnostic_item(sym::Child, child_adt.did())
        {
            match cx.tcx.parent_hir_node(expr.hir_id) {
                Node::LetStmt(local)
                    if let PatKind::Binding(_, local_id, ..) = local.pat.kind
                        && let Some(enclosing_block) = get_enclosing_block(cx, expr.hir_id) =>
                {
                    let mut vis = WaitFinder {
                        cx,
                        local_id,
                        create_id: expr.hir_id,
                        body_id: cx.tcx.hir_enclosing_body_owner(expr.hir_id),
                        state: VisitorState::WalkUpToCreate,
                        early_return: None,
                        missing_wait_branch: None,
                    };

                    let res = (
                        walk_block(&mut vis, enclosing_block),
                        vis.missing_wait_branch,
                        vis.early_return,
                    );

                    let cause = match res {
                        (Break(MaybeWait(wait_span)), _, Some(return_span)) => {
                            Cause::EarlyReturn { wait_span, return_span }
                        },
                        (Break(MaybeWait(_)), _, None) => return,
                        (Continue(()), None, _) => Cause::NeverWait,
                        (Continue(()), Some(MissingWaitBranch::MissingElse { if_span, wait_span }), _) => {
                            Cause::MissingElse { wait_span, if_span }
                        },
                        (Continue(()), Some(MissingWaitBranch::MissingWaitInBranch { branch_span, wait_span }), _) => {
                            Cause::MissingWaitInBranch { wait_span, branch_span }
                        },
                    };

                    // Don't emit a suggestion since the binding is used later
                    check(cx, expr, cause, false);
                },
                Node::LetStmt(&LetStmt { pat, .. }) if let PatKind::Wild = pat.kind => {
                    // `let _ = child;`, also dropped immediately without `wait()`ing
                    check(cx, expr, Cause::NeverWait, true);
                },
                Node::Stmt(&Stmt {
                    kind: StmtKind::Semi(_),
                    ..
                }) => {
                    // Immediately dropped. E.g. `std::process::Command::new("echo").spawn().unwrap();`
                    check(cx, expr, Cause::NeverWait, true);
                },
                _ => {},
            }
        }
    }
}

struct MaybeWait(Span);

/// A visitor responsible for finding a `wait()` call on a local variable.
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
struct WaitFinder<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    local_id: HirId,
    create_id: HirId,
    body_id: LocalDefId,
    state: VisitorState,
    early_return: Option<Span>,
    // When joining two if branches where one of them doesn't call `wait()`, stores its span for more targeted help
    // messages
    missing_wait_branch: Option<MissingWaitBranch>,
}

#[derive(PartialEq)]
enum VisitorState {
    WalkUpToCreate,
    CreateFound,
}

#[derive(Copy, Clone)]
enum MissingWaitBranch {
    MissingElse { if_span: Span, wait_span: Span },
    MissingWaitInBranch { branch_span: Span, wait_span: Span },
}

impl<'tcx> Visitor<'tcx> for WaitFinder<'_, 'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;
    type Result = ControlFlow<MaybeWait>;

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) -> Self::Result {
        if ex.hir_id == self.create_id {
            self.state = VisitorState::CreateFound;
            return Continue(());
        }

        if self.state != VisitorState::CreateFound {
            return walk_expr(self, ex);
        }

        if path_to_local_id(ex, self.local_id) {
            match self.cx.tcx.parent_hir_node(ex.hir_id) {
                Node::Stmt(Stmt {
                    kind: StmtKind::Semi(_),
                    ..
                }) => {},
                Node::Expr(expr) if let ExprKind::Field(..) = expr.kind => {},
                Node::Expr(expr) if let ExprKind::AddrOf(_, Mutability::Not, _) = expr.kind => {},
                Node::Expr(expr)
                    if let Some(fn_did) = fn_def_id(self.cx, expr)
                        && (self.cx.tcx.is_diagnostic_item(sym::child_id, fn_did)
                            || self.cx.tcx.is_diagnostic_item(sym::child_kill, fn_did)) => {},

                // Conservatively assume that all other kinds of nodes call `.wait()` somehow.
                _ => return Break(MaybeWait(ex.span)),
            }
        } else {
            match ex.kind {
                ExprKind::Ret(e) if self.cx.tcx.hir_enclosing_body_owner(ex.hir_id) == self.body_id => {
                    visit_opt!(self, visit_expr, e);
                    if self.early_return.is_none() {
                        self.early_return = Some(ex.span);
                    }

                    return Continue(());
                },
                ExprKind::If(cond, then, None) => {
                    walk_expr(self, cond)?;

                    if let Break(MaybeWait(wait_span)) = walk_expr(self, then)
                        && self.missing_wait_branch.is_none()
                    {
                        self.missing_wait_branch = Some(MissingWaitBranch::MissingElse {
                            if_span: ex.span,
                            wait_span,
                        });
                    }

                    return Continue(());
                },

                ExprKind::If(cond, then, Some(else_)) => {
                    walk_expr(self, cond)?;

                    match (walk_expr(self, then), walk_expr(self, else_)) {
                        (Continue(()), Continue(())) => {},

                        // `wait()` in one branch but nothing in the other does not count
                        (Continue(()), Break(MaybeWait(wait_span))) => {
                            if self.missing_wait_branch.is_none() {
                                self.missing_wait_branch = Some(MissingWaitBranch::MissingWaitInBranch {
                                    branch_span: ex.span.shrink_to_lo().to(then.span),
                                    wait_span,
                                });
                            }
                        },
                        (Break(MaybeWait(wait_span)), Continue(())) => {
                            if self.missing_wait_branch.is_none() {
                                self.missing_wait_branch = Some(MissingWaitBranch::MissingWaitInBranch {
                                    branch_span: then.span.shrink_to_hi().to(else_.span),
                                    wait_span,
                                });
                            }
                        },

                        // `wait()` in both branches is ok
                        (Break(MaybeWait(wait_span)), Break(MaybeWait(_))) => {
                            self.missing_wait_branch = None;
                            return Break(MaybeWait(wait_span));
                        },
                    }

                    return Continue(());
                },
                _ => {},
            }
        }

        walk_expr(self, ex)
    }

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.cx.tcx
    }
}

#[derive(Copy, Clone)]
enum Cause {
    /// No call to `wait()` at all
    NeverWait,
    /// `wait()` call exists, but not all code paths definitely lead to one due to
    /// an early return
    EarlyReturn { wait_span: Span, return_span: Span },
    /// `wait()` call exists in some if branches but not this one
    MissingWaitInBranch { wait_span: Span, branch_span: Span },
    /// `wait()` call exists in an if/then branch but it is missing an else block
    MissingElse { wait_span: Span, if_span: Span },
}

impl Cause {
    fn message(self) -> &'static str {
        match self {
            Cause::NeverWait => "spawned process is never `wait()`ed on",
            Cause::EarlyReturn { .. } | Cause::MissingWaitInBranch { .. } | Cause::MissingElse { .. } => {
                "spawned process is not `wait()`ed on in all code paths"
            },
        }
    }

    fn fallback_help(self) -> &'static str {
        match self {
            Cause::NeverWait => "consider calling `.wait()`",
            Cause::EarlyReturn { .. } | Cause::MissingWaitInBranch { .. } | Cause::MissingElse { .. } => {
                "consider calling `.wait()` in all code paths"
            },
        }
    }
}

/// This function has shared logic between the different kinds of nodes that can trigger the lint.
///
/// In particular, `let <binding> = <expr that spawns child>;` requires some custom additional logic
/// such as checking that the binding is not used in certain ways, which isn't necessary for
/// `let _ = <expr that spawns child>;`.
///
/// This checks if the program doesn't unconditionally exit after the spawn expression.
fn check<'tcx>(cx: &LateContext<'tcx>, spawn_expr: &'tcx Expr<'tcx>, cause: Cause, emit_suggestion: bool) {
    let Some(block) = get_enclosing_block(cx, spawn_expr.hir_id) else {
        return;
    };

    let mut vis = ExitPointFinder {
        state: ExitPointState::WalkUpTo(spawn_expr.hir_id),
        cx,
    };
    if let Break(ExitCallFound) = vis.visit_block(block) {
        // Visitor found an unconditional `exit()` call, so don't lint.
        return;
    }

    span_lint_and_then(cx, ZOMBIE_PROCESSES, spawn_expr.span, cause.message(), |diag| {
        match cause {
            Cause::EarlyReturn { wait_span, return_span } => {
                diag.span_note(
                    return_span,
                    "no `wait()` call exists on the code path to this early return",
                );
                diag.span_note(
                    wait_span,
                    "`wait()` call exists, but it is unreachable due to the early return",
                );
            },
            Cause::MissingWaitInBranch { wait_span, branch_span } => {
                diag.span_note(branch_span, "`wait()` is not called in this if branch");
                diag.span_note(wait_span, "`wait()` is called in the other branch");
            },
            Cause::MissingElse { if_span, wait_span } => {
                diag.span_note(
                    if_span,
                    "this if expression has a `wait()` call, but it is missing an else block",
                );
                diag.span_note(wait_span, "`wait()` called here");
            },
            Cause::NeverWait => {},
        }

        if emit_suggestion {
            diag.span_suggestion(
                spawn_expr.span.shrink_to_hi(),
                "try",
                ".wait()",
                Applicability::MaybeIncorrect,
            );
        } else {
            diag.help(cause.fallback_help());
        }

        diag.note("not doing so might leave behind zombie processes")
            .note("see https://doc.rust-lang.org/stable/std/process/struct.Child.html#warning");
    });
}

/// Checks if the given expression exits the process.
fn is_exit_expression(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    fn_def_id(cx, expr).is_some_and(|fn_did| {
        cx.tcx.is_diagnostic_item(sym::process_exit, fn_did) || cx.tcx.is_diagnostic_item(sym::process_abort, fn_did)
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
