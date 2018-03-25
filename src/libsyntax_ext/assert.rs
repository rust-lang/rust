// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::*;
use syntax::codemap::Spanned;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::ext::quote::rt::ToTokens;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax_pos::symbol::Symbol;
use syntax_pos::{Span, DUMMY_SP};

use std::cell::RefCell;
use std::rc::Rc;

macro_rules! matches {
    ($expression: expr, $($pattern:pat)|*) => (
        match $expression {
            $($pattern)|* => true,
            _ => false
        }
    );
}

pub fn expand_assert<'cx>(
    ecx: &'cx mut ExtCtxt,
    sp: Span,
    tts: &[TokenTree],
) -> Box<MacResult + 'cx> {
    let mut parser = ecx.new_parser_from_tts(tts);
    let cond_expr = panictry!(parser.parse_expr());
    let custom_msg_args = if parser.eat(&token::Comma) {
        let ts = parser.parse_tokens();
        if !ts.is_empty() {
            Some(ts)
        } else {
            None
        }
    } else {
        None
    };

    let sp = sp.with_ctxt(sp.ctxt().apply_mark(ecx.current_expansion.mark));
    MacEager::expr(if let Some(ts) = custom_msg_args {
        let panic_call = Mac_ {
            path: Path::from_ident(sp, Ident::from_str("panic")),
            tts: ts.into(),
        };
        ecx.expr_if(
            sp,
            ecx.expr(sp, ExprKind::Unary(UnOp::Not, cond_expr)),
            ecx.expr(
                sp,
                ExprKind::Mac(Spanned {
                    span: sp,
                    node: panic_call,
                }),
            ),
            None,
        )
    } else {
        Context::generate(ecx, sp, cond_expr)
    })
}

struct Context<'cx, 'a: 'cx> {
    captures: Vec<Capture>,
    ecx: &'cx ExtCtxt<'a>,
    sp: Span,
    expr_str: String,
    paths: Paths,
    tmp_match_ctr: RefCell<u32>,
}

struct Paths {
    debug_fallback_new: P<Expr>,
    try_capture: P<Expr>,
    unevaluated: Path,
    capt_unevaluated: Path,
    capt_value: P<Expr>,
}

/// A "capture", which is a intermediate value
/// that will be taken from the assertion condition.
struct Capture {
    /// Stringified expression.
    expr_str: String,
    /// The variable to store captured variable.
    var: Ident,
    /// `var`'s declaration statement.
    decl: Stmt,
    mode: BindingMode,
}

impl<'cx, 'a: 'cx> Context<'cx, 'a> {
    /// Returns the entire assertion expression which should replace the `assert!(..)` call.
    fn generate(ecx: &'cx ExtCtxt<'a>, sp: Span, expr: P<Expr>) -> P<Expr> {
        let mut generator = Self {
            captures: Vec::new(),
            ecx,
            sp,
            expr_str: unescape_printable_unicode(&pprust::expr_to_string(&expr)).escape_debug(),
            tmp_match_ctr: Default::default(),
            paths: Paths {
                debug_fallback_new: ecx.expr_path(ecx.path(
                    sp,
                    ecx.std_path(&["assert_helper", "DebugFallback", "new"]),
                )),
                try_capture: ecx.expr_path(ecx.path(
                    sp,
                    ecx.std_path(&["assert_helper", "TryCapture", "try_capture"]),
                )),
                unevaluated: ecx.path(sp, ecx.std_path(&["assert_helper", "Unevaluated"])),
                capt_unevaluated: ecx.path(
                    sp,
                    ecx.std_path(&["assert_helper", "Captured", "Unevaluated"]),
                ),
                capt_value: ecx.expr_path(ecx.path(
                    sp,
                    ecx.std_path(&["assert_helper", "Captured", "Value"]),
                )),
            },
        };
        let (cond_expr, fmt_str) = generator.scan(expr);
        let contains_lazy = cond_expr.contains_lazy;
        let expr = generator.assertion_expr(
            cond_expr,
            Action::If(
                generator.ecx.expr_tuple(generator.sp, vec![]),
                generator.panic_expr(fmt_str),
            ),
        );
        let mut stmts = generator
            .captures
            .iter()
            // If there are no branches, we can omit placeholders for by-ref captures.
            .filter(|c| contains_lazy || c.mode.is_by_value())
            .map(|c| c.decl.clone())
            .collect::<Vec<_>>();
        stmts.push(ecx.stmt_expr(expr));
        ecx.expr_block(ecx.block(sp, stmts))
    }

    /// Generates a `panic!(...)` expression.
    fn panic_expr(&self, fmt_str: String) -> P<Expr> {
        // ```
        // panic!(
        //     "assertion failed: ...",
        //     "stringified expr",
        //     assert_helper::DebugFallback::new(__capture0, "(stringified subexpr0)"),
        //     ...
        // );
        // ```
        // let mut cap_decl_stmts = Vec::with_capacity(self.captures.len());
        let fmt_str = format!("assertion failed: {{}}\nwith expansion: {}", fmt_str);
        let mut tts = Vec::new();
        macro_rules! push_tok {
            ($tts: expr, $tok: expr) => {
                $tts.push(TokenTree::Token(DUMMY_SP, $tok));
            };
        }
        push_tok!(
            tts,
            token::Literal(token::Lit::Str_(Name::intern(&fmt_str)), None)
        );
        push_tok!(tts, token::Comma);
        push_tok!(
            tts,
            token::Literal(token::Lit::Str_(Name::intern(&self.expr_str)), None)
        );
        push_tok!(tts, token::Comma);
        for cap in &self.captures {
            tts.extend(
                self.ecx
                    .expr_call(
                        self.sp,
                        self.paths.debug_fallback_new.clone(),
                        vec![
                            if cap.mode.is_by_ref() {
                                self.ecx.expr_call(
                                    self.sp,
                                    self.paths.capt_value.clone(),
                                    vec![
                                        self.ecx.expr(
                                            self.sp,
                                            ExprKind::Path(
                                                None,
                                                Path::from_ident(self.sp, cap.var),
                                            ),
                                        ),
                                    ],
                                )
                            } else {
                                self.ecx.expr(
                                    self.sp,
                                    ExprKind::Path(None, Path::from_ident(self.sp, cap.var)),
                                )
                            },
                            self.ecx.expr_lit(
                                self.sp,
                                LitKind::Str(
                                    Symbol::intern(&format!("({})", cap.expr_str)),
                                    StrStyle::Cooked,
                                ),
                            ),
                        ],
                    )
                    .to_tokens(self.ecx),
            );

            push_tok!(tts, token::Comma);
        }

        self.ecx.expr(
            self.sp,
            ExprKind::Mac(Spanned {
                span: self.sp,
                node: Mac_ {
                    path: Path::from_ident(self.sp, Ident::from_str("panic")),
                    tts: tts.into_iter().collect::<TokenStream>().into(),
                },
            }),
        )
    }

    /// Whether the expression itself (not the sub-expressions of it) needs to be captured.
    fn needs_immediate_capture(expr: &Expr) -> bool {
        match expr.node {
            ExprKind::Binary(..)
            | ExprKind::Unary(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Lit(..) => false,
            ExprKind::Paren(ref e) => Self::needs_immediate_capture(e),
            ExprKind::Block(..) | _ => true,
        }
    }

    /// Returns a `CondExpr` and a format string to interpolate captured values.
    fn scan(&mut self, expr: P<Expr>) -> (CondExpr, String) {
        let expr = expr.into_inner();
        match expr.node {
            ExprKind::Binary(op, lhs, rhs) => {
                // Comparison operators (`==`, `>=', ...) does not consume operands.
                let is_cmp = is_cmp_method(op.node);
                let ((l_ce, l_str), (r_ce, r_str)) = (
                    if is_cmp && Self::needs_immediate_capture(&lhs) {
                        self.scan_capture_expr(lhs, BindingMode::ByRef(Mutability::Immutable))
                    } else {
                        self.scan(lhs)
                    },
                    if is_cmp && Self::needs_immediate_capture(&rhs) {
                        self.scan_capture_expr(rhs, BindingMode::ByRef(Mutability::Immutable))
                    } else {
                        self.scan(rhs)
                    },
                );
                (
                    CondExpr::bin(op.node, l_ce, r_ce),
                    format!("{} {} {}", l_str, op.node.to_string(), r_str),
                )
            }
            ExprKind::Unary(op, e) => {
                // `*(capturable)`: it is captured by-ref in order not to move out from a borrowed.
                let (ce, s) = if op == UnOp::Deref && Self::needs_immediate_capture(&e) {
                    self.scan_capture_expr(e, BindingMode::ByRef(Mutability::Immutable))
                } else {
                    self.scan(e)
                };

                (
                    CondExpr::un(op, ce),
                    format!("{}{}", UnOp::to_string(op), s),
                )
            }
            ExprKind::Paren(e) => {
                let (ce, s) = self.scan(e);
                (ce, format!("({})", s))
            }
            ExprKind::AddrOf(mutbl, e) => {
                let (ce, s) = if Self::needs_immediate_capture(&e) {
                    self.scan_capture_expr(e, BindingMode::ByRef(mutbl))
                } else {
                    self.scan(e)
                };

                (
                    CondExpr::addr_of(mutbl, ce),
                    format!(
                        "&{}{}",
                        if mutbl == Mutability::Mutable {
                            "mut "
                        } else {
                            ""
                        },
                        s
                    ),
                )
            }
            ExprKind::Lit(ref lit) => (
                CondExpr::literal(lit.node.clone()),
                escape_format_string(&unescape_printable_unicode(&pprust::expr_to_string(&expr))),
            ),
            // Otherwise capture and stop recursing.
            _ => self.scan_capture_expr(P(expr), BindingMode::ByValue(Mutability::Immutable)),
        }
    }

    fn scan_capture_expr(&mut self, expr: P<Expr>, mode: BindingMode) -> (CondExpr, String) {
        if matches!(expr.node, ExprKind::Paren(..)) {
            match expr.into_inner().node {
                ExprKind::Paren(expr) => return self.scan_capture_expr(expr, mode),
                _ => unreachable!(),
            }
        }

        let capture_idx = self.captures.len();
        let capture = Ident::from_str(&format!("__capture{}", capture_idx));
        self.captures.push(Capture {
            expr_str: pprust::expr_to_string(&expr),
            var: capture,
            decl: if mode.is_by_ref() {
                // `#[allow(unused)] let __capture{} = Unevaluated;`
                // Can be unused if the variable cannot be seen from any branch.
                let allow_unused = {
                    let word = self.ecx
                        .meta_list_item_word(self.sp, Symbol::intern("unused"));
                    self.ecx.attribute(
                        self.sp,
                        self.ecx
                            .meta_list(self.sp, Symbol::intern("allow"), vec![word]),
                    )
                };
                let local = P(Local {
                    pat: self.ecx.pat_ident(self.sp, capture),
                    ty: None,
                    init: Some(self.ecx.expr_struct(
                        self.sp,
                        self.paths.unevaluated.clone(),
                        vec![],
                    )),
                    id: DUMMY_NODE_ID,
                    span: self.sp,
                    attrs: ThinVec::from(vec![allow_unused]),
                });
                Stmt {
                    id: DUMMY_NODE_ID,
                    node: StmtKind::Local(local),
                    span: self.sp,
                }
            } else {
                // `let mut __capture{} = Captured::Unevaluated;`
                self.ecx.stmt_let(
                    self.sp,
                    true,
                    capture,
                    self.ecx
                        .expr_struct(self.sp, self.paths.capt_unevaluated.clone(), vec![]),
                )
            },
            mode,
        });
        (
            CondExpr::capture(expr, capture_idx, mode),
            format!("{{{}:?}}", capture_idx + 1),
        )
    }

    /// Generates the assertion expression.
    fn assertion_expr(&self, expr: CondExpr, action: Action<'cx>) -> P<Expr> {
        if expr.contains_by_ref && expr.is_lazy_binop() {
            return match expr.node {
                CondExprKind::BinOp(BinOpKind::And, left, right) => {
                    let on_false = action.on_false(self);
                    self.assertion_expr(
                        *left,
                        Action::If(self.assertion_expr(*right, action.clone()), on_false),
                    )
                }
                CondExprKind::BinOp(BinOpKind::Or, left, right) => self.assertion_expr(
                    *left,
                    Action::If(action.on_true(self), self.assertion_expr(*right, action)),
                ),
                _ => unreachable!(),
            };
        }

        // Branching is unnecessary when there is only by-value captures,
        // since they are lazily evaluated.
        //
        // ```rust
        // // By-reference capture: eagerly evaluated
        // match &expr {
        //     __captureN => if ... { action(__captureN) }
        // }
        //
        // // By-value capture: lazily evaluated
        // if ... { action({ let tmp = expr; ... }) }
        // ```

        match expr.node {
            CondExprKind::BinOp(op, left, right) => {
                let left_by_ref = left.is_by_ref_capture();
                let right_by_ref = right.is_by_ref_capture();

                let with_left = Action::Value(Rc::new(move |left, is_evaluated| {
                    let action = action.clone();
                    let right = right.clone();
                    let process = move |left_needs_deref: bool| {
                        move |left: P<Expr>| -> P<Expr> {
                            self.assertion_expr(
                                *right,
                                Action::Value(Rc::new(move |right, _| {
                                    let left = left.clone();
                                    let left = if left_needs_deref {
                                        self.ecx.expr_unary(
                                            self.sp,
                                            UnOp::Deref,
                                            self.ecx.expr_unary(self.sp, UnOp::Deref, left),
                                        )
                                    } else {
                                        left
                                    };
                                    let expr = if is_cmp_method(op) {
                                        self.ecx.expr_binary(
                                            self.sp,
                                            op,
                                            if left_by_ref {
                                                self.ecx.expr_unary(self.sp, UnOp::Deref, left)
                                            } else {
                                                left
                                            },
                                            if right_by_ref {
                                                self.ecx.expr_unary(self.sp, UnOp::Deref, right)
                                            } else {
                                                right
                                            },
                                        )
                                    } else {
                                        self.ecx.expr_binary(self.sp, op, left, right)
                                    };
                                    action.apply(self, expr, false)
                                })),
                            )
                        }
                    };
                    if is_evaluated {
                        process(false)(left)
                    } else {
                        // If the lhs haven't been evaluated, evaluate it first.

                        // Binding would move out a borrowed content, so defer the deref.
                        let (left, left_needs_deref) = match remove_double_deref(left.clone()) {
                            Ok(e) => (e, true),
                            Err(e) => (e, false),
                        };

                        self.tmp_match(
                            left,
                            BindingMode::ByValue(Mutability::Immutable),
                            process(left_needs_deref),
                        )
                    }
                }));
                self.assertion_expr(*left, with_left)
            }
            CondExprKind::Capture(expr, idx, mode) => {
                if mode.is_by_ref() {
                    let ident = Ident::from_str(&format!("__capture{}", idx));
                    self.single_match(
                        expr,
                        ident,
                        mode,
                        action.apply(self, self.ecx.expr_ident(self.sp, ident), true),
                    )
                } else {
                    action.apply(self, self.capture_expr_by_val(expr, idx), false)
                }
            }
            CondExprKind::UnOp(op, x) => {
                let is_capture = x.is_capture();
                self.assertion_expr(
                    *x,
                    Action::Value(Rc::new(move |x, ed| {
                        let expr = self.ecx.expr_unary(self.sp, op, x);
                        let expr = if op == UnOp::Deref && is_capture {
                            // `*(capturable)`: it is captured by-ref in order not to move out
                            // from a borrowed, so deref it again to get to the correct type.
                            self.ecx.expr_unary(self.sp, UnOp::Deref, expr)
                        } else {
                            expr
                        };
                        action.apply(self, expr, ed)
                    })),
                )
            }
            CondExprKind::AddrOf(m, e) => {
                let is_capture = e.is_capture();
                self.assertion_expr(
                    *e,
                    Action::Value(Rc::new(move |val, ed| {
                        let val = if is_capture {
                            self.ecx.expr_unary(self.sp, UnOp::Deref, val)
                        } else {
                            val
                        };
                        let val = match m {
                            Mutability::Immutable => self.ecx.expr_addr_of(self.sp, val),
                            Mutability::Mutable => self.ecx.expr_mut_addr_of(self.sp, val),
                        };
                        action.apply(self, val, ed)
                    })),
                )
            }
            CondExprKind::Literal(lit) => action.apply(self, self.ecx.expr_lit(self.sp, lit), true),
        }
    }

    fn single_match(
        &self,
        expr: P<Expr>,
        ident: Ident,
        mode: BindingMode,
        inner_expr: P<Expr>,
    ) -> P<Expr> {
        self.ecx.expr_match(
            self.sp,
            expr,
            vec![
                self.ecx.arm(
                    self.sp,
                    vec![self.ecx.pat_ident_binding_mode(self.sp, ident, mode)],
                    inner_expr,
                ),
            ],
        )
    }

    /// Modifies the expression to capture the value of it.
    fn capture_expr_by_val(&self, expr: P<Expr>, idx: usize) -> P<Expr> {
        let capture = Ident::from_str(&format!("__capture{}", idx));

        // `expr` →
        // `{ let __tmp = expr; __tmp.try_capture(&mut __captureN); __tmp }`
        let tmp = Ident::from_str("__tmp");
        self.ecx.expr_block(self.ecx.block(
            self.sp,
            vec![
                self.ecx.stmt_let(self.sp, false, tmp, expr),
                self.ecx.stmt_semi(self.ecx.expr_call(
                    self.sp,
                    self.paths.try_capture.clone(),
                    vec![
                        self.ecx.expr_addr_of(
                            self.sp,
                            self.ecx.expr_path(Path::from_ident(self.sp, tmp)),
                        ),
                        self.ecx.expr_mut_addr_of(
                            self.sp,
                            self.ecx.expr_path(Path::from_ident(self.sp, capture)),
                        ),
                    ],
                )),
                self.ecx.stmt_expr(self.ecx.expr_path(Path::from_ident(self.sp, tmp))),
            ],
        ))
    }

    fn tmp_match<F: FnOnce(P<Expr>) -> P<Expr>>(
        &self,
        expr: P<Expr>,
        mode: BindingMode,
        f: F,
    ) -> P<Expr> {
        let tmp_idx = {
            let mut c = self.tmp_match_ctr.borrow_mut();
            *c += 1;
            *c - 1
        };
        let ident = Ident::from_str(&format!("__tmp{}", tmp_idx));
        self.single_match(expr, ident, mode, f(self.ecx.expr_ident(self.sp, ident)))
    }
}

#[derive(Clone)]
struct CondExpr {
    node: CondExprKind,
    /// True if `||` or `&&` are contained in this tree.
    contains_lazy: bool,
    /// True if by-value captures are contained in this tree.
    contains_by_ref: bool,
}

/// Part of `ExprKind` we take care of, plus `Capture` that describes how to capture.
#[derive(Clone)]
enum CondExprKind {
    BinOp(BinOpKind, Box<CondExpr>, Box<CondExpr>),
    UnOp(UnOp, Box<CondExpr>),
    AddrOf(Mutability, Box<CondExpr>),
    Capture(P<Expr>, usize, BindingMode),
    Literal(LitKind),
}

impl CondExpr {
    fn bin(op: BinOpKind, left: CondExpr, right: CondExpr) -> Self {
        CondExpr {
            contains_lazy: left.contains_lazy || right.contains_lazy || op.lazy(),
            contains_by_ref: left.contains_by_ref || right.contains_by_ref || is_cmp_method(op),
            node: CondExprKind::BinOp(op, left.into(), right.into()),
        }
    }

    fn un(op: UnOp, expr: CondExpr) -> Self {
        CondExpr {
            contains_lazy: expr.contains_lazy,
            contains_by_ref: expr.contains_by_ref,
            node: CondExprKind::UnOp(op, expr.into()),
        }
    }

    fn addr_of(m: Mutability, expr: CondExpr) -> Self {
        CondExpr {
            contains_lazy: expr.contains_lazy,
            contains_by_ref: expr.contains_by_ref,
            node: CondExprKind::AddrOf(m, expr.into()),
        }
    }

    fn capture(expr: P<Expr>, idx: usize, mode: BindingMode) -> Self {
        CondExpr {
            contains_lazy: false,
            contains_by_ref: mode.is_by_ref(),
            node: CondExprKind::Capture(expr, idx, mode),
        }
    }

    fn literal(lit: LitKind) -> Self {
        CondExpr {
            contains_lazy: false,
            contains_by_ref: false,
            node: CondExprKind::Literal(lit),
        }
    }

    fn is_capture(&self) -> bool {
        matches!(self.node, CondExprKind::Capture(..))
    }

    fn is_by_ref_capture(&self) -> bool {
        matches!(
            self.node,
            CondExprKind::Capture(_, _, BindingMode::ByRef(_))
        )
    }

    fn is_lazy_binop(&self) -> bool {
        if let CondExprKind::BinOp(op, ..) = self.node {
            op.lazy()
        } else {
            false
        }
    }
}

/// What to do next based on the result of the condition.
#[derive(Clone)]
enum Action<'cx> {
    /// `If(then, else)`.
    If(P<Expr>, P<Expr>),
    /// An arbitrary expression that uses the value.
    /// `IsEvaluated` argument is used to reduce unnecesary
    /// temporary bindings for forcing evaluation.
    Value(Rc<Fn(P<Expr>, IsEvaluated) -> P<Expr> + 'cx>),
}

type IsEvaluated = bool;

impl<'cx> Action<'cx> {
    fn on_true(&self, acx: &Context) -> P<Expr> {
        match *self {
            Action::If(ref t, _) => t.clone(),
            Action::Value(ref fun) => fun(acx.ecx.expr_lit(acx.sp, LitKind::Bool(true)), true),
        }
    }

    fn on_false(&self, acx: &Context) -> P<Expr> {
        match *self {
            Action::If(_, ref f) => f.clone(),
            Action::Value(ref fun) => fun(acx.ecx.expr_lit(acx.sp, LitKind::Bool(false)), true),
        }
    }

    fn apply(&self, acx: &Context, value: P<Expr>, is_evaluated: bool) -> P<Expr> {
        match *self {
            Action::If(ref t, ref f) => acx.ecx.expr_if(acx.sp, value, t.clone(), Some(f.clone())),
            Action::Value(ref fun) => fun(value, is_evaluated),
        }
    }
}

/// Removes the outermost double dereferences. e.g. `**x` => `Ok(x)`, `*y` => `Err(*y)`
fn remove_double_deref(e: P<Expr>) -> Result<P<Expr>, P<Expr>> {
    let available = if let ExprKind::Unary(UnOp::Deref, ref inner) = e.node {
        matches!(inner.node, ExprKind::Unary(UnOp::Deref, _))
    } else {
        false
    };
    if available {
        if let ExprKind::Unary(_, inner) = e.into_inner().node {
            if let ExprKind::Unary(_, inner2) = inner.into_inner().node {
                return Ok(inner2);
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }

    Err(e)
}

fn is_cmp_method(op: BinOpKind) -> bool {
    matches!(
        op,
        BinOpKind::Eq | BinOpKind::Ne | BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt
            | BinOpKind::Ge
    )
}

/// Escapes a string for use as a formatting string.
fn escape_format_string(s: &str) -> String {
    let mut res = String::with_capacity(s.len());
    for c in s.chars() {
        res.extend(c.escape_debug());
        match c {
            '{' | '}' => res.push(c),
            _ => {}
        }
    }
    res
}

#[test]
fn test_escape_format_string() {
    assert!(escape_format_string("foo{}") == "foo{{}}");
}

/// Unescapes the escaped unicodes (`\u{...}`) that are printable.
fn unescape_printable_unicode(mut s: &str) -> String {
    use std::{char, u32};

    let mut res = String::with_capacity(s.len());

    loop {
        if let Some(start) = s.find(r"\u{") {
            res.push_str(&s[0..start]);
            s = &s[start..];
            s.find('}')
                .and_then(|end| {
                    let v = u32::from_str_radix(&s[3..end], 16).ok()?;
                    let c = char::from_u32(v)?;
                    // Escape unprintable characters.
                    res.extend(c.escape_debug());
                    s = &s[end + 1..];
                    Some(())
                })
                .expect("lexer should have rejected invalid escape sequences");
        } else {
            res.push_str(s);
            return res;
        }
    }
}

#[test]
fn test_unescape_printable_unicode() {
    assert!(unescape_printable_unicode(r"\u{2603}\n\u{0}") == r"☃\n\u{0}");
}
