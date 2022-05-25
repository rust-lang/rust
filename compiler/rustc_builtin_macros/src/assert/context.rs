use rustc_ast::{
    attr,
    ptr::P,
    token,
    tokenstream::{DelimSpan, TokenStream, TokenTree},
    BinOpKind, BorrowKind, Expr, ExprKind, ItemKind, MacArgs, MacCall, MacDelimiter, Mutability,
    Path, PathSegment, Stmt, StructRest, UnOp, UseTree, UseTreeKind, DUMMY_NODE_ID,
};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::ExtCtxt;
use rustc_span::{
    symbol::{sym, Ident, Symbol},
    Span,
};

pub(super) struct Context<'cx, 'a> {
    // An optimization.
    //
    // Elements that aren't consumed (PartialEq, PartialOrd, ...) can be copied **after** the
    // `assert!` expression fails rather than copied on-the-fly.
    best_case_captures: Vec<Stmt>,
    // Top-level `let captureN = Capture::new()` statements
    capture_decls: Vec<Capture>,
    cx: &'cx ExtCtxt<'a>,
    // Top-level `let __local_bindN = &expr` statements
    local_bind_decls: Vec<Stmt>,
    // Used to avoid capturing duplicated paths
    //
    // ```rust
    // let a = 1i32;
    // assert!(add(a, a) == 3);
    // ```
    paths: FxHashSet<String>,
    span: Span,
}

impl<'cx, 'a> Context<'cx, 'a> {
    pub(super) fn new(cx: &'cx ExtCtxt<'a>, span: Span) -> Self {
        Self {
            best_case_captures: <_>::default(),
            capture_decls: <_>::default(),
            cx,
            local_bind_decls: <_>::default(),
            paths: <_>::default(),
            span,
        }
    }

    /// Builds the whole `assert!` expression.
    ///
    /// {
    ///    use ::core::asserting::{ ... };
    ///
    ///    let mut __capture0 = Capture::new();
    ///    ...
    ///    ...
    ///    ...
    ///
    ///    if !{
    ///       ...
    ///       ...
    ///       ...
    ///    } {
    ///        panic!(
    ///            "Assertion failed: ... \n With expansion: ...",
    ///            __capture0,
    ///            ...
    ///            ...
    ///            ...
    ///        );
    ///    }
    /// }
    pub(super) fn build(mut self, mut cond_expr: P<Expr>, panic_path: Path) -> P<Expr> {
        let expr_str = pprust::expr_to_string(&cond_expr);
        let mut fmt_str = String::new();
        self.manage_cond_expr(&mut cond_expr, &mut fmt_str, true);
        let initial_imports = self.build_initial_imports();
        let panic = self.build_panic(&expr_str, &fmt_str, panic_path);
        let cond_expr_with_unlikely = self.build_unlikely(cond_expr);

        let Self { best_case_captures, capture_decls, cx, local_bind_decls, .. } = self;

        let mut assert_then_stmts = Vec::with_capacity(2);
        assert_then_stmts.extend(best_case_captures);
        assert_then_stmts.push(self.cx.stmt_expr(panic));
        let assert_then = self.cx.block(self.span, assert_then_stmts);

        let mut stmts = Vec::with_capacity(4);
        stmts.push(initial_imports);
        stmts.extend(capture_decls.into_iter().map(|c| c.decl));
        stmts.extend(local_bind_decls);
        stmts.push(cx.stmt_expr(
            cx.expr(self.span, ExprKind::If(cond_expr_with_unlikely, assert_then, None)),
        ));
        cx.expr_block(cx.block(self.span, stmts))
    }

    /// Initial **trait** imports
    ///
    /// use ::core::asserting::{ ... };
    fn build_initial_imports(&self) -> Stmt {
        self.cx.stmt_item(
            self.span,
            self.cx.item(
                self.span,
                Ident::empty(),
                vec![self.cx.attribute(attr::mk_list_item(
                    Ident::new(sym::allow, self.span),
                    vec![attr::mk_nested_word_item(Ident::new(sym::unused_imports, self.span))],
                ))],
                ItemKind::Use(UseTree {
                    prefix: self.cx.path(self.span, self.cx.std_path(&[sym::asserting])),
                    kind: UseTreeKind::Nested(vec![
                        (
                            UseTree {
                                prefix: self.cx.path(
                                    self.span,
                                    vec![Ident::with_dummy_span(sym::TryCaptureGeneric)],
                                ),
                                kind: UseTreeKind::Simple(None, DUMMY_NODE_ID, DUMMY_NODE_ID),
                                span: self.span,
                            },
                            DUMMY_NODE_ID,
                        ),
                        (
                            UseTree {
                                prefix: self.cx.path(
                                    self.span,
                                    vec![Ident::with_dummy_span(sym::TryCapturePrintable)],
                                ),
                                kind: UseTreeKind::Simple(None, DUMMY_NODE_ID, DUMMY_NODE_ID),
                                span: self.span,
                            },
                            DUMMY_NODE_ID,
                        ),
                    ]),
                    span: self.span,
                }),
            ),
        )
    }

    /// The necessary custom `panic!(...)` expression.
    ///
    /// panic!(
    ///     "Assertion failed: ... \n With expansion: ...",
    ///     __capture0,
    ///     ...
    /// );
    fn build_panic(&self, expr_str: &str, fmt_str: &str, panic_path: Path) -> P<Expr> {
        let escaped_expr_str = escape_to_fmt(expr_str);
        let initial = [
            TokenTree::token(
                token::Literal(token::Lit {
                    kind: token::LitKind::Str,
                    symbol: Symbol::intern(&if fmt_str.is_empty() {
                        format!("Assertion failed: {escaped_expr_str}")
                    } else {
                        format!("Assertion failed: {escaped_expr_str}\nWith captures:\n{fmt_str}")
                    }),
                    suffix: None,
                }),
                self.span,
            ),
            TokenTree::token(token::Comma, self.span),
        ];
        let captures = self.capture_decls.iter().flat_map(|cap| {
            [
                TokenTree::token(token::Ident(cap.ident.name, false), cap.ident.span),
                TokenTree::token(token::Comma, self.span),
            ]
        });
        self.cx.expr(
            self.span,
            ExprKind::MacCall(MacCall {
                path: panic_path,
                args: P(MacArgs::Delimited(
                    DelimSpan::from_single(self.span),
                    MacDelimiter::Parenthesis,
                    initial.into_iter().chain(captures).collect::<TokenStream>(),
                )),
                prior_type_ascription: None,
            }),
        )
    }

    /// Takes the conditional expression of `assert!` and then wraps it inside `unlikely`
    fn build_unlikely(&self, cond_expr: P<Expr>) -> P<Expr> {
        let unlikely_path = self.cx.std_path(&[sym::intrinsics, sym::unlikely]);
        self.cx.expr_call(
            self.span,
            self.cx.expr_path(self.cx.path(self.span, unlikely_path)),
            vec![self.cx.expr(self.span, ExprKind::Unary(UnOp::Not, cond_expr))],
        )
    }

    /// Recursive function called until `cond_expr` and `fmt_str` are fully modified.
    ///
    /// See [Self::manage_initial_capture] and [Self::manage_try_capture]
    fn manage_cond_expr(&mut self, expr: &mut P<Expr>, fmt_str: &mut String, is_consumed: bool) {
        match (*expr).kind {
            ExprKind::AddrOf(_, mutability, ref mut local_expr) => {
                self.manage_cond_expr(local_expr, fmt_str, matches!(mutability, Mutability::Mut));
            }
            ExprKind::Array(ref mut local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr, fmt_str, true);
                }
            }
            ExprKind::Binary(ref op, ref mut lhs, ref mut rhs) => {
                let local_is_consumed = matches!(
                    op.node,
                    BinOpKind::Add
                        | BinOpKind::And
                        | BinOpKind::BitAnd
                        | BinOpKind::BitOr
                        | BinOpKind::BitXor
                        | BinOpKind::Div
                        | BinOpKind::Mul
                        | BinOpKind::Or
                        | BinOpKind::Rem
                        | BinOpKind::Shl
                        | BinOpKind::Shr
                        | BinOpKind::Sub
                );
                self.manage_cond_expr(lhs, fmt_str, local_is_consumed);
                self.manage_cond_expr(rhs, fmt_str, local_is_consumed);
            }
            ExprKind::Call(_, ref mut local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr, fmt_str, true);
                }
            }
            ExprKind::Cast(ref mut local_expr, _) => {
                self.manage_cond_expr(local_expr, fmt_str, true);
            }
            ExprKind::Index(ref mut prefix, ref mut suffix) => {
                self.manage_cond_expr(prefix, fmt_str, true);
                self.manage_cond_expr(suffix, fmt_str, true);
            }
            ExprKind::MethodCall(_, ref mut local_exprs, _) => {
                for local_expr in local_exprs.iter_mut().skip(1) {
                    self.manage_cond_expr(local_expr, fmt_str, true);
                }
            }
            ExprKind::Path(_, ref path) => {
                let string = pprust::path_to_string(&path);
                self.manage_initial_capture(expr, string, fmt_str, is_consumed);
            }
            ExprKind::Paren(ref mut local_expr) => {
                self.manage_cond_expr(local_expr, fmt_str, true);
            }
            ExprKind::Range(ref mut prefix, ref mut suffix, _) => {
                if let Some(ref mut elem) = prefix {
                    self.manage_cond_expr(elem, fmt_str, true);
                }
                if let Some(ref mut elem) = suffix {
                    self.manage_cond_expr(elem, fmt_str, true);
                }
            }
            ExprKind::Repeat(ref mut local_expr, ref mut elem) => {
                self.manage_cond_expr(local_expr, fmt_str, true);
                self.manage_cond_expr(&mut elem.value, fmt_str, true);
            }
            ExprKind::Struct(ref mut elem) => {
                for field in &mut elem.fields {
                    self.manage_cond_expr(&mut field.expr, fmt_str, true);
                }
                if let StructRest::Base(ref mut local_expr) = elem.rest {
                    self.manage_cond_expr(local_expr, fmt_str, true);
                }
            }
            ExprKind::Tup(ref mut local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr, fmt_str, true);
                }
            }
            ExprKind::Unary(un_op, ref mut local_expr) => {
                self.manage_cond_expr(local_expr, fmt_str, matches!(un_op, UnOp::Neg | UnOp::Not));
            }
            // Expressions that are not worth or can not be captured.
            //
            // Full list instead of `_` to catch possible future inclusions and to
            // sync the `rfc-2011-nicer-assert-messages/all-expr-kinds.rs` test.
            ExprKind::Assign(_, _, _)
            | ExprKind::AssignOp(_, _, _)
            | ExprKind::Async(_, _, _)
            | ExprKind::Await(_)
            | ExprKind::Block(_, _)
            | ExprKind::Box(_)
            | ExprKind::Break(_, _)
            | ExprKind::Closure(_, _, _, _, _, _)
            | ExprKind::ConstBlock(_)
            | ExprKind::Continue(_)
            | ExprKind::Err
            | ExprKind::Field(_, _)
            | ExprKind::ForLoop(_, _, _, _)
            | ExprKind::If(_, _, _)
            | ExprKind::InlineAsm(_)
            | ExprKind::Let(_, _, _)
            | ExprKind::Lit(_)
            | ExprKind::Loop(_, _)
            | ExprKind::MacCall(_)
            | ExprKind::Match(_, _)
            | ExprKind::Ret(_)
            | ExprKind::Try(_)
            | ExprKind::TryBlock(_)
            | ExprKind::Type(_, _)
            | ExprKind::Underscore
            | ExprKind::While(_, _, _)
            | ExprKind::Yeet(_)
            | ExprKind::Yield(_) => {}
        }
    }

    /// Pushes the top-level declarations and modifies `expr` to try capturing variables.
    ///
    /// `fmt_str`, the formatting string used for debugging, is constructed to show the possible
    /// captured variables.
    fn manage_initial_capture(
        &mut self,
        expr: &mut P<Expr>,
        expr_string: String,
        fmt_str: &mut String,
        is_consumed: bool,
    ) {
        if self.paths.contains(&expr_string) {
            return;
        } else {
            fmt_str.push_str("  ");
            fmt_str.push_str(&expr_string);
            fmt_str.push_str(" = {:?}\n");
            let _ = self.paths.insert(expr_string);
        }
        let curr_capture_idx = self.capture_decls.len();
        let capture_string = format!("__capture{curr_capture_idx}");
        let ident = Ident::new(Symbol::intern(&capture_string), self.span);
        let init_std_path = self.cx.std_path(&[sym::asserting, sym::Capture, sym::new]);
        let init = self.cx.expr_call(
            self.span,
            self.cx.expr_path(self.cx.path(self.span, init_std_path)),
            vec![],
        );
        let capture = Capture { decl: self.cx.stmt_let(self.span, true, ident, init), ident };
        self.capture_decls.push(capture);
        self.manage_try_capture(ident, curr_capture_idx, expr, is_consumed);
    }

    /// Tries to copy `__local_bindN` into `__captureN`.
    ///
    /// *{
    ///    (&Wrapper(__local_bindN)).try_capture(&mut __captureN);
    ///    __local_bindN
    /// }
    fn manage_try_capture(
        &mut self,
        capture: Ident,
        curr_capture_idx: usize,
        expr: &mut P<Expr>,
        is_consumed: bool,
    ) {
        let local_bind_string = format!("__local_bind{curr_capture_idx}");
        let local_bind = Ident::new(Symbol::intern(&local_bind_string), self.span);
        self.local_bind_decls.push(self.cx.stmt_let(
            self.span,
            false,
            local_bind,
            self.cx.expr_addr_of(self.span, expr.clone()),
        ));
        let wrapper = self.cx.expr_call(
            self.span,
            self.cx.expr_path(
                self.cx.path(self.span, self.cx.std_path(&[sym::asserting, sym::Wrapper])),
            ),
            vec![self.cx.expr_path(Path::from_ident(local_bind))],
        );
        let try_capture_call = self
            .cx
            .stmt_expr(expr_method_call(
                self.cx,
                PathSegment {
                    args: None,
                    id: DUMMY_NODE_ID,
                    ident: Ident::new(sym::try_capture, self.span),
                },
                vec![
                    expr_paren(self.cx, self.span, self.cx.expr_addr_of(self.span, wrapper)),
                    expr_addr_of_mut(
                        self.cx,
                        self.span,
                        self.cx.expr_path(Path::from_ident(capture)),
                    ),
                ],
                self.span,
            ))
            .add_trailing_semicolon();
        let local_bind_path = self.cx.expr_path(Path::from_ident(local_bind));
        let rslt = if is_consumed {
            let ret = self.cx.stmt_expr(local_bind_path);
            self.cx.expr_block(self.cx.block(self.span, vec![try_capture_call, ret]))
        } else {
            self.best_case_captures.push(try_capture_call);
            local_bind_path
        };
        *expr = self.cx.expr_deref(self.span, rslt);
    }
}

/// Information about a captured element.
///
/// All the following fields will use `let a = 3; assert!(a > 1);` as an example.
#[derive(Debug)]
struct Capture {
    // Generated indexed `Capture` statement regarding `a`.
    //
    // `let __capture{} = Capture::new();`
    decl: Stmt,
    // The name of the generated indexed `Capture` variable.
    //
    // `__capture{}`
    ident: Ident,
}

/// Escapes to use as a formatting string.
fn escape_to_fmt(s: &str) -> String {
    let mut rslt = String::with_capacity(s.len());
    for c in s.chars() {
        rslt.extend(c.escape_debug());
        match c {
            '{' | '}' => rslt.push(c),
            _ => {}
        }
    }
    rslt
}

fn expr_addr_of_mut(cx: &ExtCtxt<'_>, sp: Span, e: P<Expr>) -> P<Expr> {
    cx.expr(sp, ExprKind::AddrOf(BorrowKind::Ref, Mutability::Mut, e))
}

fn expr_method_call(
    cx: &ExtCtxt<'_>,
    path: PathSegment,
    args: Vec<P<Expr>>,
    span: Span,
) -> P<Expr> {
    cx.expr(span, ExprKind::MethodCall(path, args, span))
}

fn expr_paren(cx: &ExtCtxt<'_>, sp: Span, e: P<Expr>) -> P<Expr> {
    cx.expr(sp, ExprKind::Paren(e))
}
