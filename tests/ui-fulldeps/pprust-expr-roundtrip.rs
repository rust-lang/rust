//@ run-pass
//@ ignore-cross-compile
//@ aux-crate: parser=parser.rs
//@ edition: 2021

// The general idea of this test is to enumerate all "interesting" expressions and check that
// `parse(print(e)) == e` for all `e`. Here's what's interesting, for the purposes of this test:
//
// 1. The test focuses on expression nesting, because interactions between different expression
//    types are harder to test manually than single expression types in isolation.
//
// 2. The test only considers expressions of at most two nontrivial nodes. So it will check `x +
//    x` and `x + (x - x)` but not `(x * x) + (x - x)`. The assumption here is that the correct
//    handling of an expression might depend on the expression's parent, but doesn't depend on its
//    siblings or any more distant ancestors.
//
// 3. The test only checks certain expression kinds. The assumption is that similar expression
//    types, such as `if` and `while` or `+` and `-`, will be handled identically in the printer
//    and parser. So if all combinations of exprs involving `if` work correctly, then combinations
//    using `while`, `if let`, and so on will likely work as well.

#![feature(rustc_private)]

extern crate rustc_ast;
extern crate rustc_ast_pretty;
extern crate rustc_parse;
extern crate rustc_session;
extern crate rustc_span;
extern crate thin_vec;

// Necessary to pull in object code as the rest of the rustc crates are shipped only as rmeta
// files.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use parser::parse_expr;
use rustc_ast::mut_visit::MutVisitor;
use rustc_ast::*;
use rustc_ast_pretty::pprust;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::DUMMY_SP;
use thin_vec::{thin_vec, ThinVec};

// Helper functions for building exprs
fn expr(kind: ExprKind) -> Box<Expr> {
    Box::new(Expr { id: DUMMY_NODE_ID, kind, span: DUMMY_SP, attrs: AttrVec::new(), tokens: None })
}

fn make_x() -> Box<Expr> {
    let seg = PathSegment::from_ident(Ident::from_str("x"));
    let path = Path { segments: thin_vec![seg], span: DUMMY_SP, tokens: None };
    expr(ExprKind::Path(None, path))
}

/// Iterate over exprs of depth up to `depth`. The goal is to explore all "interesting"
/// combinations of expression nesting. For example, we explore combinations using `if`, but not
/// `while` or `match`, since those should print and parse in much the same way as `if`.
fn iter_exprs(depth: usize, f: &mut dyn FnMut(Box<Expr>)) {
    if depth == 0 {
        f(make_x());
        return;
    }

    let mut g = |e| f(expr(e));

    for kind in 0..=18 {
        match kind {
            0 => iter_exprs(depth - 1, &mut |e| g(ExprKind::Call(e, thin_vec![]))),
            1 => {
                let seg = PathSegment::from_ident(Ident::from_str("x"));
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::MethodCall(Box::new(MethodCall {
                        seg: seg.clone(),
                        receiver: e,
                        args: thin_vec![make_x()],
                        span: DUMMY_SP,
                    })))
                });
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::MethodCall(Box::new(MethodCall {
                        seg: seg.clone(),
                        receiver: make_x(),
                        args: thin_vec![e],
                        span: DUMMY_SP,
                    })))
                });
            }
            2..=7 => {
                let op = Spanned {
                    span: DUMMY_SP,
                    node: match kind {
                        2 => BinOpKind::Add,
                        3 => BinOpKind::Mul,
                        4 => BinOpKind::Shl,
                        5 => BinOpKind::And,
                        6 => BinOpKind::Or,
                        7 => BinOpKind::Lt,
                        _ => unreachable!(),
                    },
                };
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Binary(op, e, make_x())));
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Binary(op, make_x(), e)));
            }
            8 => {
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Unary(UnOp::Deref, e)));
            }
            9 => {
                let block = Box::new(Block {
                    stmts: ThinVec::new(),
                    id: DUMMY_NODE_ID,
                    rules: BlockCheckMode::Default,
                    span: DUMMY_SP,
                    tokens: None,
                });
                iter_exprs(depth - 1, &mut |e| g(ExprKind::If(e, block.clone(), None)));
            }
            10 => {
                let decl = Box::new(FnDecl {
                    inputs: thin_vec![],
                    output: FnRetTy::Default(DUMMY_SP),
                });
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::Closure(Box::new(Closure {
                        binder: ClosureBinder::NotPresent,
                        capture_clause: CaptureBy::Value { move_kw: DUMMY_SP },
                        constness: Const::No,
                        coroutine_kind: None,
                        movability: Movability::Movable,
                        fn_decl: decl.clone(),
                        body: e,
                        fn_decl_span: DUMMY_SP,
                        fn_arg_span: DUMMY_SP,
                    })))
                });
            }
            11 => {
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Assign(e, make_x(), DUMMY_SP)));
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Assign(make_x(), e, DUMMY_SP)));
            }
            12 => {
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Field(e, Ident::from_str("f"))));
            }
            13 => {
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::Range(Some(e), Some(make_x()), RangeLimits::HalfOpen))
                });
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::Range(Some(make_x()), Some(e), RangeLimits::HalfOpen))
                });
            }
            14 => {
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::AddrOf(BorrowKind::Ref, Mutability::Not, e))
                });
            }
            15 => {
                g(ExprKind::Ret(None));
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Ret(Some(e))));
            }
            16 => {
                let path = Path::from_ident(Ident::from_str("S"));
                g(ExprKind::Struct(Box::new(StructExpr {
                    qself: None,
                    path,
                    fields: thin_vec![],
                    rest: StructRest::Base(make_x()),
                })));
            }
            17 => {
                iter_exprs(depth - 1, &mut |e| g(ExprKind::Try(e)));
            }
            18 => {
                let pat = Box::new(Pat {
                    id: DUMMY_NODE_ID,
                    kind: PatKind::Wild,
                    span: DUMMY_SP,
                    tokens: None,
                });
                iter_exprs(depth - 1, &mut |e| {
                    g(ExprKind::Let(pat.clone(), e, DUMMY_SP, Recovered::No))
                })
            }
            _ => panic!("bad counter value in iter_exprs"),
        }
    }
}

// Folders for manipulating the placement of `Paren` nodes. See below for why this is needed.

/// `MutVisitor` that removes all `ExprKind::Paren` nodes.
struct RemoveParens;

impl MutVisitor for RemoveParens {
    fn visit_expr(&mut self, e: &mut Expr) {
        match e.kind.clone() {
            ExprKind::Paren(inner) => *e = *inner,
            _ => {}
        };
        mut_visit::walk_expr(self, e);
    }
}

/// `MutVisitor` that inserts `ExprKind::Paren` nodes around every `Expr`.
struct AddParens;

impl MutVisitor for AddParens {
    fn visit_expr(&mut self, e: &mut Expr) {
        mut_visit::walk_expr(self, e);
        let expr = std::mem::replace(e, Expr::dummy());

        e.kind = ExprKind::Paren(Box::new(expr));
    }
}

fn main() {
    rustc_span::create_default_session_globals_then(|| run());
}

fn run() {
    let psess = ParseSess::new(vec![rustc_parse::DEFAULT_LOCALE_RESOURCE]);

    iter_exprs(2, &mut |mut e| {
        // If the pretty printer is correct, then `parse(print(e))` should be identical to `e`,
        // modulo placement of `Paren` nodes.
        let printed = pprust::expr_to_string(&e);
        println!("printed: {}", printed);

        // Ignore expressions with chained comparisons that fail to parse
        if let Some(mut parsed) = parse_expr(&psess, &printed) {
            // We want to know if `parsed` is structurally identical to `e`, ignoring trivial
            // differences like placement of `Paren`s or the exact ranges of node spans.
            // Unfortunately, there is no easy way to make this comparison. Instead, we add `Paren`s
            // everywhere we can, then pretty-print. This should give an unambiguous representation
            // of each `Expr`, and it bypasses nearly all of the parenthesization logic, so we
            // aren't relying on the correctness of the very thing we're testing.
            RemoveParens.visit_expr(&mut e);
            AddParens.visit_expr(&mut e);
            let text1 = pprust::expr_to_string(&e);
            RemoveParens.visit_expr(&mut parsed);
            AddParens.visit_expr(&mut parsed);
            let text2 = pprust::expr_to_string(&parsed);
            assert!(
                text1 == text2,
                "exprs are not equal:\n  e =      {:?}\n  parsed = {:?}",
                text1,
                text2
            );
        }
    });
}
