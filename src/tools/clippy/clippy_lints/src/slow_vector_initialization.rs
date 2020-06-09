use crate::utils::sugg::Sugg;
use crate::utils::{get_enclosing_block, match_qpath, span_lint_and_then, SpanlessEq};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::intravisit::{walk_block, walk_expr, walk_stmt, NestedVisitorMap, Visitor};
use rustc_hir::{BindingAnnotation, Block, Expr, ExprKind, HirId, PatKind, QPath, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, Lint};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// **What it does:** Checks slow zero-filled vector initialization
    ///
    /// **Why is this bad?** These structures are non-idiomatic and less efficient than simply using
    /// `vec![0; len]`.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # use core::iter::repeat;
    /// # let len = 4;
    ///
    /// // Bad
    /// let mut vec1 = Vec::with_capacity(len);
    /// vec1.resize(len, 0);
    ///
    /// let mut vec2 = Vec::with_capacity(len);
    /// vec2.extend(repeat(0).take(len));
    ///
    /// // Good
    /// let mut vec1 = vec![0; len];
    /// let mut vec2 = vec![0; len];
    /// ```
    pub SLOW_VECTOR_INITIALIZATION,
    perf,
    "slow vector initialization"
}

declare_lint_pass!(SlowVectorInit => [SLOW_VECTOR_INITIALIZATION]);

/// `VecAllocation` contains data regarding a vector allocated with `with_capacity` and then
/// assigned to a variable. For example, `let mut vec = Vec::with_capacity(0)` or
/// `vec = Vec::with_capacity(0)`
struct VecAllocation<'tcx> {
    /// Symbol of the local variable name
    variable_name: Symbol,

    /// Reference to the expression which allocates the vector
    allocation_expr: &'tcx Expr<'tcx>,

    /// Reference to the expression used as argument on `with_capacity` call. This is used
    /// to only match slow zero-filling idioms of the same length than vector initialization.
    len_expr: &'tcx Expr<'tcx>,
}

/// Type of slow initialization
enum InitializationType<'tcx> {
    /// Extend is a slow initialization with the form `vec.extend(repeat(0).take(..))`
    Extend(&'tcx Expr<'tcx>),

    /// Resize is a slow initialization with the form `vec.resize(.., 0)`
    Resize(&'tcx Expr<'tcx>),
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for SlowVectorInit {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        // Matches initialization on reassignements. For example: `vec = Vec::with_capacity(100)`
        if_chain! {
            if let ExprKind::Assign(ref left, ref right, _) = expr.kind;

            // Extract variable name
            if let ExprKind::Path(QPath::Resolved(_, ref path)) = left.kind;
            if let Some(variable_name) = path.segments.get(0);

            // Extract len argument
            if let Some(ref len_arg) = Self::is_vec_with_capacity(right);

            then {
                let vi = VecAllocation {
                    variable_name: variable_name.ident.name,
                    allocation_expr: right,
                    len_expr: len_arg,
                };

                Self::search_initialization(cx, vi, expr.hir_id);
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt<'_>) {
        // Matches statements which initializes vectors. For example: `let mut vec = Vec::with_capacity(10)`
        if_chain! {
            if let StmtKind::Local(ref local) = stmt.kind;
            if let PatKind::Binding(BindingAnnotation::Mutable, .., variable_name, None) = local.pat.kind;
            if let Some(ref init) = local.init;
            if let Some(ref len_arg) = Self::is_vec_with_capacity(init);

            then {
                let vi = VecAllocation {
                    variable_name: variable_name.name,
                    allocation_expr: init,
                    len_expr: len_arg,
                };

                Self::search_initialization(cx, vi, stmt.hir_id);
            }
        }
    }
}

impl SlowVectorInit {
    /// Checks if the given expression is `Vec::with_capacity(..)`. It will return the expression
    /// of the first argument of `with_capacity` call if it matches or `None` if it does not.
    fn is_vec_with_capacity<'tcx>(expr: &Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
        if_chain! {
            if let ExprKind::Call(ref func, ref args) = expr.kind;
            if let ExprKind::Path(ref path) = func.kind;
            if match_qpath(path, &["Vec", "with_capacity"]);
            if args.len() == 1;

            then {
                return Some(&args[0]);
            }
        }

        None
    }

    /// Search initialization for the given vector
    fn search_initialization<'tcx>(cx: &LateContext<'_, 'tcx>, vec_alloc: VecAllocation<'tcx>, parent_node: HirId) {
        let enclosing_body = get_enclosing_block(cx, parent_node);

        if enclosing_body.is_none() {
            return;
        }

        let mut v = VectorInitializationVisitor {
            cx,
            vec_alloc,
            slow_expression: None,
            initialization_found: false,
        };

        v.visit_block(enclosing_body.unwrap());

        if let Some(ref allocation_expr) = v.slow_expression {
            Self::lint_initialization(cx, allocation_expr, &v.vec_alloc);
        }
    }

    fn lint_initialization<'tcx>(
        cx: &LateContext<'_, 'tcx>,
        initialization: &InitializationType<'tcx>,
        vec_alloc: &VecAllocation<'_>,
    ) {
        match initialization {
            InitializationType::Extend(e) | InitializationType::Resize(e) => Self::emit_lint(
                cx,
                e,
                vec_alloc,
                "slow zero-filling initialization",
                SLOW_VECTOR_INITIALIZATION,
            ),
        };
    }

    fn emit_lint<'tcx>(
        cx: &LateContext<'_, 'tcx>,
        slow_fill: &Expr<'_>,
        vec_alloc: &VecAllocation<'_>,
        msg: &str,
        lint: &'static Lint,
    ) {
        let len_expr = Sugg::hir(cx, vec_alloc.len_expr, "len");

        span_lint_and_then(cx, lint, slow_fill.span, msg, |diag| {
            diag.span_suggestion(
                vec_alloc.allocation_expr.span,
                "consider replace allocation with",
                format!("vec![0; {}]", len_expr),
                Applicability::Unspecified,
            );
        });
    }
}

/// `VectorInitializationVisitor` searches for unsafe or slow vector initializations for the given
/// vector.
struct VectorInitializationVisitor<'a, 'tcx> {
    cx: &'a LateContext<'a, 'tcx>,

    /// Contains the information.
    vec_alloc: VecAllocation<'tcx>,

    /// Contains the slow initialization expression, if one was found.
    slow_expression: Option<InitializationType<'tcx>>,

    /// `true` if the initialization of the vector has been found on the visited block.
    initialization_found: bool,
}

impl<'a, 'tcx> VectorInitializationVisitor<'a, 'tcx> {
    /// Checks if the given expression is extending a vector with `repeat(0).take(..)`
    fn search_slow_extend_filling(&mut self, expr: &'tcx Expr<'_>) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.kind;
            if let ExprKind::Path(ref qpath_subj) = args[0].kind;
            if match_qpath(&qpath_subj, &[&*self.vec_alloc.variable_name.as_str()]);
            if path.ident.name == sym!(extend);
            if let Some(ref extend_arg) = args.get(1);
            if self.is_repeat_take(extend_arg);

            then {
                self.slow_expression = Some(InitializationType::Extend(expr));
            }
        }
    }

    /// Checks if the given expression is resizing a vector with 0
    fn search_slow_resize_filling(&mut self, expr: &'tcx Expr<'_>) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.kind;
            if let ExprKind::Path(ref qpath_subj) = args[0].kind;
            if match_qpath(&qpath_subj, &[&*self.vec_alloc.variable_name.as_str()]);
            if path.ident.name == sym!(resize);
            if let (Some(ref len_arg), Some(fill_arg)) = (args.get(1), args.get(2));

            // Check that is filled with 0
            if let ExprKind::Lit(ref lit) = fill_arg.kind;
            if let LitKind::Int(0, _) = lit.node;

            // Check that len expression is equals to `with_capacity` expression
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_alloc.len_expr);

            then {
                self.slow_expression = Some(InitializationType::Resize(expr));
            }
        }
    }

    /// Returns `true` if give expression is `repeat(0).take(...)`
    fn is_repeat_take(&self, expr: &Expr<'_>) -> bool {
        if_chain! {
            if let ExprKind::MethodCall(ref take_path, _, ref take_args) = expr.kind;
            if take_path.ident.name == sym!(take);

            // Check that take is applied to `repeat(0)`
            if let Some(ref repeat_expr) = take_args.get(0);
            if Self::is_repeat_zero(repeat_expr);

            // Check that len expression is equals to `with_capacity` expression
            if let Some(ref len_arg) = take_args.get(1);
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_alloc.len_expr);

            then {
                return true;
            }
        }

        false
    }

    /// Returns `true` if given expression is `repeat(0)`
    fn is_repeat_zero(expr: &Expr<'_>) -> bool {
        if_chain! {
            if let ExprKind::Call(ref fn_expr, ref repeat_args) = expr.kind;
            if let ExprKind::Path(ref qpath_repeat) = fn_expr.kind;
            if match_qpath(&qpath_repeat, &["repeat"]);
            if let Some(ref repeat_arg) = repeat_args.get(0);
            if let ExprKind::Lit(ref lit) = repeat_arg.kind;
            if let LitKind::Int(0, _) = lit.node;

            then {
                return true
            }
        }

        false
    }
}

impl<'a, 'tcx> Visitor<'tcx> for VectorInitializationVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_stmt(&mut self, stmt: &'tcx Stmt<'_>) {
        if self.initialization_found {
            match stmt.kind {
                StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => {
                    self.search_slow_extend_filling(expr);
                    self.search_slow_resize_filling(expr);
                },
                _ => (),
            }

            self.initialization_found = false;
        } else {
            walk_stmt(self, stmt);
        }
    }

    fn visit_block(&mut self, block: &'tcx Block<'_>) {
        if self.initialization_found {
            if let Some(ref s) = block.stmts.get(0) {
                self.visit_stmt(s)
            }

            self.initialization_found = false;
        } else {
            walk_block(self, block);
        }
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'_>) {
        // Skip all the expressions previous to the vector initialization
        if self.vec_alloc.allocation_expr.hir_id == expr.hir_id {
            self.initialization_found = true;
        }

        walk_expr(self, expr);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}
