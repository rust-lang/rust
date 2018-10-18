// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::rustc::hir::intravisit::{walk_expr, NestedVisitorMap, Visitor};
use crate::rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use crate::rustc::{declare_tool_lint, lint_array};
use crate::rustc::hir::*;
use if_chain::if_chain;
use crate::syntax_pos::symbol::Symbol;
use crate::syntax::ast::{LitKind, NodeId};
use crate::syntax::source_map::Span;
use crate::utils::{match_qpath, span_lint_and_then, SpanlessEq};
use crate::utils::get_enclosing_block;
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
    "slow or unsafe vector initialization"
}

#[derive(Copy, Clone, Default)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(SLOW_VECTOR_INITIALIZATION)
    }
}

/// VecInitialization contains data regarding a vector initialized with `with_capacity` and then
/// assigned to a variable. For example, `let mut vec = Vec::with_capacity(0)` or
/// `vec = Vec::with_capacity(0)`
struct VecInitialization<'tcx> {
    /// Symbol of the local variable name
    variable_name: Symbol,

    /// Reference to the expression which initializes the vector
    initialization_expr: &'tcx Expr,

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
            if let Some(ref len_arg) = Pass::is_vec_with_capacity(right);

            then {
                let vi = VecInitialization {
                    variable_name: variable_name.ident.name,
                    initialization_expr: right,
                    len_expr: len_arg,
                };

                Pass::search_slow_zero_filling(cx, vi, expr.id, expr.span);
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
            if let Some(ref len_arg) = Pass::is_vec_with_capacity(init);

            then {
                let vi = VecInitialization {
                    variable_name: variable_name.name,
                    initialization_expr: init,
                    len_expr: len_arg,
                };

                Pass::search_slow_zero_filling(cx, vi, stmt.node.id(), stmt.span);
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

    /// Search for slow zero filling vector initialization for the given vector
    fn search_slow_zero_filling<'tcx>(
        cx: &LateContext<'_, 'tcx>,
        vec_initialization: VecInitialization<'tcx>,
        parent_node: NodeId,
        parent_span: Span
    ) {
        let enclosing_body = get_enclosing_block(cx, parent_node);

        if enclosing_body.is_none() {
            return;
        }

        let mut v = SlowInitializationVisitor {
            cx,
            vec_ini: vec_initialization,
            slow_expression: None,
            initialization_found: false,
        };

        v.visit_block(enclosing_body.unwrap());

        if let Some(ref repeat_expr) = v.slow_expression {
            span_lint_and_then(
                cx,
                SLOW_VECTOR_INITIALIZATION,
                parent_span,
                "detected slow zero-filling initialization",
                |db| {
                    db.span_suggestion_with_applicability(v.vec_ini.initialization_expr.span, "consider replacing with", "vec![0; ..]".to_string(), Applicability::Unspecified);

                    match repeat_expr {
                        InitializationType::Extend(e) => {
                            db.span_note(e.span, "extended here with .. 0");
                        },
                        InitializationType::Resize(e) => {
                            db.span_note(e.span, "resize here with .. 0");
                        }
                    }
                }
            );
        }
    }
}

/// SlowInitializationVisitor searches for slow zero filling vector initialization, for the given
/// vector.
struct SlowInitializationVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,

    /// Contains the information
    vec_ini: VecInitialization<'tcx>,

    /// Contains, if found, the slow initialization expression
    slow_expression: Option<InitializationType<'tcx>>,

    /// true if the initialization of the vector has been found on the visited block
    initialization_found: bool,
}

impl<'a, 'tcx> SlowInitializationVisitor<'a, 'tcx> {
    /// Checks if the given expression is extending a vector with `repeat(0).take(..)`
    fn search_slow_extend_filling(&mut self, expr: &'tcx Expr) {
        if_chain! {
            if self.initialization_found;
            if let ExprKind::MethodCall(ref path, _, ref args) = expr.node;
            if let ExprKind::Path(ref qpath_subj) = args[0].node;
            if match_qpath(&qpath_subj, &[&self.vec_ini.variable_name.to_string()]);
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
            if match_qpath(&qpath_subj, &[&self.vec_ini.variable_name.to_string()]);
            if path.ident.name == "resize";
            if let (Some(ref len_arg), Some(fill_arg)) = (args.get(1), args.get(2));

            // Check that is filled with 0
            if let ExprKind::Lit(ref lit) = fill_arg.node;
            if let LitKind::Int(0, _) = lit.node;

            // Check that len expression is equals to `with_capacity` expression
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_ini.len_expr);

            then {
                self.slow_expression = Some(InitializationType::Resize(expr));
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
            if SpanlessEq::new(self.cx).eq_expr(len_arg, self.vec_ini.len_expr);

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

impl<'a, 'tcx> Visitor<'tcx> for SlowInitializationVisitor<'a, 'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr) {
        // Stop the search if we already found a slow zero-filling initialization
        if self.slow_expression.is_some() {
            return
        }

        // Skip all the expressions previous to the vector initialization
        if self.vec_ini.initialization_expr.id == expr.id {
            self.initialization_found = true;
        }
        
        self.search_slow_extend_filling(expr);
        self.search_slow_resize_filling(expr);

        walk_expr(self, expr);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}
