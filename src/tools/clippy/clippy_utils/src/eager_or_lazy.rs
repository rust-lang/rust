//! Utilities for evaluating whether eagerly evaluated expressions can be made lazy and vice versa.
//!
//! Things to consider:
//!  - has the expression side-effects?
//!  - is the expression computationally expensive?
//!
//! See lints:
//!  - unnecessary-lazy-evaluations
//!  - or-fun-call
//!  - option-if-let-else

use crate::is_ctor_or_promotable_const_function;
use crate::ty::is_type_diagnostic_item;
use rustc_hir::def::{DefKind, Res};

use rustc_hir::intravisit;
use rustc_hir::intravisit::{NestedVisitorMap, Visitor};

use rustc_hir::{Block, Expr, ExprKind, Path, QPath};
use rustc_lint::LateContext;
use rustc_middle::hir::map::Map;
use rustc_span::sym;

/// Is the expr pure (is it free from side-effects)?
/// This function is named so to stress that it isn't exhaustive and returns FNs.
fn identify_some_pure_patterns(expr: &Expr<'_>) -> bool {
    match expr.kind {
        ExprKind::Lit(..) | ExprKind::ConstBlock(..) | ExprKind::Path(..) | ExprKind::Field(..) => true,
        ExprKind::AddrOf(_, _, addr_of_expr) => identify_some_pure_patterns(addr_of_expr),
        ExprKind::Tup(tup_exprs) => tup_exprs.iter().all(identify_some_pure_patterns),
        ExprKind::Struct(_, fields, expr) => {
            fields.iter().all(|f| identify_some_pure_patterns(f.expr)) && expr.map_or(true, identify_some_pure_patterns)
        },
        ExprKind::Call(
            &Expr {
                kind:
                    ExprKind::Path(QPath::Resolved(
                        _,
                        Path {
                            res: Res::Def(DefKind::Ctor(..) | DefKind::Variant, ..),
                            ..
                        },
                    )),
                ..
            },
            args,
        ) => args.iter().all(identify_some_pure_patterns),
        ExprKind::Block(
            &Block {
                stmts,
                expr: Some(expr),
                ..
            },
            _,
        ) => stmts.is_empty() && identify_some_pure_patterns(expr),
        ExprKind::Box(..)
        | ExprKind::Array(..)
        | ExprKind::Call(..)
        | ExprKind::MethodCall(..)
        | ExprKind::Binary(..)
        | ExprKind::Unary(..)
        | ExprKind::Let(..)
        | ExprKind::Cast(..)
        | ExprKind::Type(..)
        | ExprKind::DropTemps(..)
        | ExprKind::Loop(..)
        | ExprKind::If(..)
        | ExprKind::Match(..)
        | ExprKind::Closure(..)
        | ExprKind::Block(..)
        | ExprKind::Assign(..)
        | ExprKind::AssignOp(..)
        | ExprKind::Index(..)
        | ExprKind::Break(..)
        | ExprKind::Continue(..)
        | ExprKind::Ret(..)
        | ExprKind::InlineAsm(..)
        | ExprKind::LlvmInlineAsm(..)
        | ExprKind::Repeat(..)
        | ExprKind::Yield(..)
        | ExprKind::Err => false,
    }
}

/// Identify some potentially computationally expensive patterns.
/// This function is named so to stress that its implementation is non-exhaustive.
/// It returns FNs and FPs.
fn identify_some_potentially_expensive_patterns<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    // Searches an expression for method calls or function calls that aren't ctors
    struct FunCallFinder<'a, 'tcx> {
        cx: &'a LateContext<'tcx>,
        found: bool,
    }

    impl<'a, 'tcx> intravisit::Visitor<'tcx> for FunCallFinder<'a, 'tcx> {
        type Map = Map<'tcx>;

        fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
            let call_found = match &expr.kind {
                // ignore enum and struct constructors
                ExprKind::Call(..) => !is_ctor_or_promotable_const_function(self.cx, expr),
                ExprKind::Index(obj, _) => {
                    let ty = self.cx.typeck_results().expr_ty(obj);
                    is_type_diagnostic_item(self.cx, ty, sym::hashmap_type)
                        || is_type_diagnostic_item(self.cx, ty, sym::BTreeMap)
                },
                ExprKind::MethodCall(..) => true,
                _ => false,
            };

            if call_found {
                self.found |= true;
            }

            if !self.found {
                intravisit::walk_expr(self, expr);
            }
        }

        fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
            NestedVisitorMap::None
        }
    }

    let mut finder = FunCallFinder { cx, found: false };
    finder.visit_expr(expr);
    finder.found
}

pub fn is_eagerness_candidate<'a, 'tcx>(cx: &'a LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    !identify_some_potentially_expensive_patterns(cx, expr) && identify_some_pure_patterns(expr)
}

pub fn is_lazyness_candidate<'a, 'tcx>(cx: &'a LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> bool {
    identify_some_potentially_expensive_patterns(cx, expr)
}
