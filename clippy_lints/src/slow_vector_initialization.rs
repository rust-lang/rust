// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass, Lint};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc::hir::*;
use if_chain::if_chain;
use crate::syntax_pos::symbol::Symbol;
use crate::syntax::ast::{LitKind, NodeId};
use crate::utils::{match_qpath, span_lint_and_then, SpanlessEq, get_enclosing_block};
use crate::utils::sugg::Sugg;
use crate::rustc_errors::{Applicability};

/// **What it does:** Checks slow zero-filled vector initialization
///
/// **Why is this bad?** This structures are non-idiomatic and less efficient than simply using
/// `vec![len; 0]`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let mut vec1 = Vec::with_capacity(len);
/// vec1.resize(len, 0);
///
/// let mut vec2 = Vec::with_capacity(len);
/// vec2.extend(repeat(0).take(len))
/// ```
declare_clippy_lint! {
    pub SLOW_VECTOR_INITIALIZATION,
    perf,
    "slow vector initialization"
}

/// **What it does:** Checks unsafe vector initialization
///
/// **Why is this bad?** Changing the length of a vector may expose uninitialized memory, which
/// can lead to memory safety issues
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let mut vec1 = Vec::with_capacity(len);
/// unsafe { vec1.set_len(len); }
/// ```
declare_clippy_lint! {
    pub UNSAFE_VECTOR_INITIALIZATION,
    correctness,
    "unsafe vector initialization"
}

#[derive(Copy, Clone, Default)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            SLOW_VECTOR_INITIALIZATION,
            UNSAFE_VECTOR_INITIALIZATION,
        )
    }
}

/// `VecAllocation` contains data regarding a vector allocated with `with_capacity` and then
/// assigned to a variable. For example, `let mut vec = Vec::with_capacity(0)` or
/// `vec = Vec::with_capacity(0)`
struct VecAllocation<'tcx> {
    /// Symbol of the local variable name
    variable_name: Symbol,

    /// Reference to the expression which allocates the vector
    allocation_expr: &'tcx Expr,

    /// Reference to the expression used as argument on `with_capacity` call. This is used
    /// to only match slow zero-filling idioms of the same length than vector initialization.
    len_expr: &'tcx Expr,
}

/// Type of slow initialization
enum InitializationType<'tcx> {
    /// Extend is a slow initialization with the form `vec.extend(repeat(0).take(..))`
    Extend(&'tcx Expr),

    /// Resize is a slow initialization with the form `vec.resize(.., 0)`
    Resize(&'tcx Expr),

    /// UnsafeSetLen is a slow initialization with the form `vec.set_len(..)`
    UnsafeSetLen(&'tcx Expr),
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Pass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        // Matches initialization on reassignements. For example: `vec = Vec::with_capacity(100)`
        if_chain! {
            if let ExprKind::Assign(ref left, ref right) = expr.node;

            // Extract variable name
            if let ExprKind::Path(QPath::Resolved(_, ref path)) = left.node;
            if let Some(variable_name) = path.segments.get(0);

            // Extract len argument
            if let Some(ref len_arg) = Self::is_vec_with_capacity(right);

            then {
                let vi = VecAllocation {
                    variable_name: variable_name.ident.name,
                    allocation_expr: right,
                    len_expr: len_arg,
                };

                Self::search_initialization(cx, vi, expr.id);
            }
        }
    }

    fn check_stmt(&mut self, cx: &LateContext<'a, 'tcx>, stmt: &'tcx Stmt) {
        // Matches statements which initializes vectors. For example: `let mut vec = Vec::with_capacity(10)`
        if_chain! {
            if let StmtKind::Decl(ref decl, _) = stmt.node;
            if let DeclKind::Local(ref local) = decl.node;
            if let PatKind::Binding(BindingAnnotation::Mutable, _, variable_name, None) = local.pat.node;
            if let Some(ref init) = local.init;
            if let Some(ref len_arg) = Self::is_vec_with_capacity(init);

            then {
                let vi = VecAllocation {
                    variable_name: variable_name.name,
                    allocation_expr: init,
                    len_expr: len_arg,
                };

                Self::search_initialization(cx, vi, stmt.node.id());
            }
        }
    }
}

impl Pass {
    /// Checks if the given expression is `Vec::with_capacity(..)`. It will return the expression
    /// of the first argument of `with_capacity` call if it matches or `None` if it does not.
    fn is_vec_with_capacity(expr: &Expr) -> Option<&Expr> {
        if_chain! {
            if let ExprKind::Call(ref func, ref args) = expr.node;
            if let ExprKind::Path(ref path) = func.node;
            if match_qpath(path, &["Vec", "with_capacity"]);
            if args.len() == 1;

            then {
                return Some(&args[0]);
            }
        }

        None
    }

    /// Search initialization for the given vector
    fn search_initialization<'tcx>(
        cx: &LateContext<'_, 'tcx>,
        vec_alloc: VecAllocation<'tcx>,
        parent_node: NodeId
    ) {
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

    fn lint_initialization<'tcx>(cx: &LateContext<'_, 'tcx>, initialization: &InitializationType<'tcx>, vec_alloc: &VecAllocation<'_>) {
        match initialization {
            InitializationType::UnsafeSetLen(e) =>
                Self::emit_lint(
                    cx,
                    e,
                    vec_alloc,
                    "unsafe vector initialization",
                    UNSAFE_VECTOR_INITIALIZATION
                ),

            InitializationType::Extend(e) |
            InitializationType::Resize(e) =>
                Self::emit_lint(
                    cx,
                    e,
                    vec_alloc,
                    "slow zero-filling initialization",
                    SLOW_VECTOR_INITIALIZATION
                )
        };
    }

    fn emit_lint<'tcx>(
        cx: &LateContext<'_, 'tcx>,
        slow_fill: &Expr,
        vec_alloc: &VecAllocation<'_>,
        msg: &str,
        lint: &'static Lint
    ) {
        let len_expr = Sugg::hir(cx, vec_alloc.len_expr, "len");

        span_lint_and_then(
            cx,
            lint,
            slow_fill.span,
            msg,
            |db| {
                db.span_suggestion_with_applicability(
                    vec_alloc.allocation_expr.span,
                    "consider replace allocation with",
                    format!("vec![0; {}]", len_expr),
                    Applicability::Unspecified
                );
            }
        );
    }
}

/// `VectorInitializationVisitor` searches for unsafe or slow vector initializations for the given
/// vector.
struct VectorInitializationVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,

    /// Contains the information
    vec_alloc: VecAllocation<'tcx>,

    /// Contains, if found, the slow initialization expression
    slow_expression: Option<InitializationType<'tcx>>,

    /// true if the initialization of the vector has been found on the visited block
    initialization_found: bool,
}

impl<'a, 'tcx> VectorInitializationVisitor<'a, 'tcx> {
    /// Checks if the given expression is extending a vector with `repeat(0).take(..)`
    fn search_slow_extend_filling(&mut self, expr: &'tcx Expr) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            if let ExprKind::Path(ref qpath_subj) = args[0].node;
            if match_qpath(&qpath_subj, &[&self.vec_alloc.variable_name.to_string()]);
            if path.ident.name == "extend";
            if let Some(ref extend_arg) = args.get(1);
            if self.is_repeat_take(extend_arg);

            then {
                self.slow_expression = Some(InitializationType::Extend(expr));
            }
        }
    }

    /// Checks if the given expression is resizing a vector with 0
    fn search_slow_resize_filling(&mut self, expr: &'tcx Expr) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            if let ExprKind::Path(ref qpath_subj) = args[0].node;
            if match_qpath(&qpath_subj, &[&self.vec_alloc.variable_name.to_string()]);
            if path.ident.name == "resize";
            if let (Some(ref len_arg), Some(fill_arg)) = (args.get(1), args.get(2));

            // Check that is filled with 0
            if let ExprKind::Lit(ref lit) = fill_arg.node;
            if let LitKind::Int(0, _) = lit.node;

            // Check that len expression is equals to `with_capacity` expression
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_alloc.len_expr);

            then {
                self.slow_expression = Some(InitializationType::Resize(expr));
            }
        }
    }

    /// Checks if the given expression is using `set_len` to initialize the vector
    fn search_unsafe_set_len(&mut self, expr: &'tcx Expr) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            if let ExprKind::Path(ref qpath_subj) = args[0].node;
            if match_qpath(&qpath_subj, &[&self.vec_alloc.variable_name.to_string()]);
            if path.ident.name == "set_len";
            if let Some(ref len_arg) = args.get(1);

            // Check that len expression is equals to `with_capacity` expression
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_alloc.len_expr);

            then {
                self.slow_expression = Some(InitializationType::UnsafeSetLen(expr));
            }
        }
    }

    /// Returns `true` if give expression is `repeat(0).take(...)`
    fn is_repeat_take(&self, expr: &Expr) -> bool {
        if_chain! {
            if let ExprKind::MethodCall(ref take_path, _, ref take_args) = expr.node;
            if take_path.ident.name == "take";

            // Check that take is applied to `repeat(0)`
            if let Some(ref repeat_expr) = take_args.get(0);
            if self.is_repeat_zero(repeat_expr);

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
    fn is_repeat_zero(&self, expr: &Expr) -> bool {
        if_chain! {
            if let ExprKind::Call(ref fn_expr, ref repeat_args) = expr.node;
            if let ExprKind::Path(ref qpath_repeat) = fn_expr.node;
            if match_qpath(&qpath_repeat, &["repeat"]);
            if let Some(ref repeat_arg) = repeat_args.get(0);
            if let ExprKind::Lit(ref lit) = repeat_arg.node;
            if let LitKind::Int(0, _) = lit.node;

            then {
                return true
            }
        }

        false
    }
}

impl<'a, 'tcx> Visitor<'tcx> for VectorInitializationVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        // Stop the search if we already found a slow zero-filling initialization
        if self.slow_expression.is_some() {
            return
        }

        // Skip all the expressions previous to the vector initialization
        if self.vec_alloc.allocation_expr.id == expr.id {
            self.initialization_found = true;
        }
        
        self.search_slow_extend_filling(expr);
        self.search_slow_resize_filling(expr);
        self.search_unsafe_set_len(expr);

        walk_expr(self, expr);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
