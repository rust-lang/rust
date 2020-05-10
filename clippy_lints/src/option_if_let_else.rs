use crate::utils;
use crate::utils::sugg::Sugg;
use crate::utils::{match_type, paths, span_lint_and_sugg};
use if_chain::if_chain;

use rustc_errors::Applicability;
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::map::Map;
use rustc_session::{declare_lint_pass, declare_tool_lint};

use std::marker::PhantomData;

declare_clippy_lint! {
    /// **What it does:**
    /// Lints usage of  `if let Some(v) = ... { y } else { x }` which is more
    /// idiomatically done with `Option::map_or` (if the else bit is a simple
    /// expression) or `Option::map_or_else` (if the else bit is a longer
    /// block).
    ///
    /// **Why is this bad?**
    /// Using the dedicated functions of the Option type is clearer and
    /// more concise than an if let expression.
    ///
    /// **Known problems:**
    /// This lint uses whether the block is just an expression or if it has
    /// more statements to decide whether to use `Option::map_or` or
    /// `Option::map_or_else`. If you have a single expression which calls
    /// an expensive function, then it would be more efficient to use
    /// `Option::map_or_else`, but this lint would suggest `Option::map_or`.
    ///
    /// Also, this lint uses a deliberately conservative metric for checking
    /// if the inside of either body contains breaks or continues which will
    /// cause it to not suggest a fix if either block contains a loop with
    /// continues or breaks contained within the loop.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     5
    /// };
    /// let _ = if let Some(foo) = optional {
    ///     foo
    /// } else {
    ///     let y = do_complicated_function();
    ///     y*y
    /// };
    /// ```
    ///
    /// should be
    ///
    /// ```rust
    /// # let optional: Option<u32> = Some(0);
    /// let _ = optional.map_or(5, |foo| foo);
    /// let _ = optional.map_or_else(||{
    ///     let y = do_complicated_function;
    ///     y*y
    /// }, |foo| foo);
    /// ```
    pub OPTION_IF_LET_ELSE,
    style,
    "reimplementation of Option::map_or"
}

declare_lint_pass!(OptionIfLetElse => [OPTION_IF_LET_ELSE]);

/// Returns true iff the given expression is the result of calling Result::ok
fn is_result_ok(cx: &LateContext<'_, '_>, expr: &'_ Expr<'_>) -> bool {
    if_chain! {
        if let ExprKind::MethodCall(ref path, _, &[ref receiver]) = &expr.kind;
        if path.ident.name.to_ident_string() == "ok";
        if match_type(cx, &cx.tables.expr_ty(&receiver), &paths::RESULT);
        then {
            true
        } else {
            false
        }
    }
}

/// A struct containing information about occurences of the
/// `if let Some(..) = .. else` construct that this lint detects.
struct OptionIfLetElseOccurence {
    option: String,
    method_sugg: String,
    some_expr: String,
    none_expr: String,
    wrap_braces: bool,
}

struct ReturnBreakContinueVisitor<'tcx> {
    seen_return_break_continue: bool,
    phantom_data: PhantomData<&'tcx bool>,
}
impl<'tcx> ReturnBreakContinueVisitor<'tcx> {
    fn new() -> ReturnBreakContinueVisitor<'tcx> {
        ReturnBreakContinueVisitor {
            seen_return_break_continue: false,
            phantom_data: PhantomData,
        }
    }
}
impl<'tcx> Visitor<'tcx> for ReturnBreakContinueVisitor<'tcx> {
    type Map = Map<'tcx>;
    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
        if self.seen_return_break_continue {
            // No need to look farther if we've already seen one of them
            return;
        }
        match &ex.kind {
            ExprKind::Ret(..) | ExprKind::Break(..) | ExprKind::Continue(..) => {
                self.seen_return_break_continue = true;
            },
            // Something special could be done here to handle while or for loop
            // desugaring, as this will detect a break if there's a while loop
            // or a for loop inside the expression.
            _ => {
                rustc_hir::intravisit::walk_expr(self, ex);
            },
        }
    }
}

fn contains_return_break_continue<'tcx>(expression: &'tcx Expr<'tcx>) -> bool {
    let mut recursive_visitor: ReturnBreakContinueVisitor<'tcx> = ReturnBreakContinueVisitor::new();
    recursive_visitor.visit_expr(expression);
    recursive_visitor.seen_return_break_continue
}

/// If this expression is the option if let/else construct we're detecting, then
/// this function returns an OptionIfLetElseOccurence struct with details if
/// this construct is found, or None if this construct is not found.
fn detect_option_if_let_else<'a>(cx: &LateContext<'_, 'a>, expr: &'a Expr<'a>) -> Option<OptionIfLetElseOccurence> {
    if_chain! {
        if !utils::in_macro(expr.span); // Don't lint macros, because it behaves weirdly
        if let ExprKind::Match(let_body, arms, MatchSource::IfLetDesugar{contains_else_clause: true}) = &expr.kind;
        if arms.len() == 2;
        // if type_is_option(cx, &cx.tables.expr_ty(let_body).kind);
        if !is_result_ok(cx, let_body); // Don't lint on Result::ok because a different lint does it already
        if let PatKind::TupleStruct(struct_qpath, &[inner_pat], _) = &arms[0].pat.kind;
        if utils::match_qpath(struct_qpath, &paths::OPTION_SOME);
        if let PatKind::Binding(bind_annotation, _, id, _) = &inner_pat.kind;
        if !contains_return_break_continue(arms[0].body);
        if !contains_return_break_continue(arms[1].body);
        then {
            let (capture_mut, capture_ref, capture_ref_mut) = match bind_annotation {
                BindingAnnotation::Unannotated => (false, false, false),
                BindingAnnotation::Mutable => (true, false, false),
                BindingAnnotation::Ref => (false, true, false),
                BindingAnnotation::RefMut => (false, false, true),
            };
            let some_body = if let ExprKind::Block(Block { stmts: statements, expr: Some(expr), .. }, _)
                = &arms[0].body.kind {
                if let &[] = &statements {
                    expr
                } else {
                    &arms[0].body
                }
            } else {
                return None;
            };
            let (none_body, method_sugg) = if let ExprKind::Block(Block { stmts: statements, expr: Some(expr), .. }, _)
                = &arms[1].body.kind {
                if let &[] = &statements {
                    (expr, "map_or")
                } else {
                    (&arms[1].body, "map_or_else")
                }
            } else {
                return None;
            };
            let capture_name = id.name.to_ident_string();
            let wrap_braces = utils::get_enclosing_block(cx, expr.hir_id).map_or(false, |parent| {
                if_chain! {
                    if let Some(Expr { kind: ExprKind::Match(
                                _,
                                arms,
                                MatchSource::IfDesugar{contains_else_clause: true}
                                    | MatchSource::IfLetDesugar{contains_else_clause: true}),
                                .. } ) = parent.expr;
                    if expr.hir_id == arms[1].body.hir_id;
                    then {
                        true
                    } else {
                        false
                    }
                }
            });
            let (parens_around_option, as_ref, as_mut, let_body) = match &let_body.kind {
                ExprKind::Call(..)
                        | ExprKind::MethodCall(..)
                        | ExprKind::Loop(..)
                        | ExprKind::Match(..)
                        | ExprKind::Block(..)
                        | ExprKind::Field(..)
                        | ExprKind::Path(_)
                    => (false, capture_ref, capture_ref_mut, let_body),
                ExprKind::Unary(UnOp::UnDeref, expr) => (false, capture_ref, capture_ref_mut, expr),
                ExprKind::AddrOf(_, mutability, expr) => (false, mutability == &Mutability::Not, mutability == &Mutability::Mut, expr),
                _ => (true, capture_ref, capture_ref_mut, let_body),
            };
            Some(OptionIfLetElseOccurence {
                option: format!("{}{}{}{}", if parens_around_option { "(" } else { "" }, Sugg::hir(cx, let_body, ".."), if parens_around_option { ")" } else { "" }, if as_mut { ".as_mut()" } else if as_ref { ".as_ref()" } else { "" }),
                method_sugg: format!("{}", method_sugg),
                some_expr: format!("|{}{}{}| {}", if false { "ref " } else { "" }, if capture_mut { "mut " } else { "" }, capture_name, Sugg::hir(cx, some_body, "..")),
                none_expr: format!("{}{}", if method_sugg == "map_or" { "" } else { "|| " }, Sugg::hir(cx, none_body, "..")),
                wrap_braces,
            })
        } else {
            None
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for OptionIfLetElse {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr<'_>) {
        if let Some(detection) = detect_option_if_let_else(cx, expr) {
            span_lint_and_sugg(
                cx,
                OPTION_IF_LET_ELSE,
                expr.span,
                format!("use Option::{} instead of an if let/else", detection.method_sugg).as_str(),
                "try",
                format!(
                    "{}{}.{}({}, {}){}",
                    if detection.wrap_braces { "{ " } else { "" },
                    detection.option,
                    detection.method_sugg,
                    detection.none_expr,
                    detection.some_expr,
                    if detection.wrap_braces { " }" } else { "" },
                ),
                Applicability::MachineApplicable,
            );
        }
    }
}
