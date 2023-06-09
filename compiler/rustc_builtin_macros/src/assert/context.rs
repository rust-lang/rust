use rustc_ast::{
    ptr::P,
    token,
    tokenstream::{DelimSpan, TokenStream, TokenTree},
    BinOpKind, BorrowKind, DelimArgs, Expr, ExprKind, ItemKind, MacCall, MacDelimiter, MethodCall,
    Mutability, Path, PathSegment, Stmt, StructRest, UnOp, UseTree, UseTreeKind, DUMMY_NODE_ID,
};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_expand::base::ExtCtxt;
use rustc_span::{
    symbol::{sym, Ident, Symbol},
    Span,
};
use thin_vec::{thin_vec, ThinVec};

pub(super) struct Context<'cx, 'a> {
    // An optimization.
    //
    // Elements that aren't consumed (PartialEq, PartialOrd, ...) can be copied **after** the
    // `assert!` expression fails rather than copied on-the-fly.
    best_case_captures: Vec<Stmt>,
    // Top-level `let captureN = Capture::new()` statements
    capture_decls: Vec<Capture>,
    cx: &'cx ExtCtxt<'a>,
    // Formatting string used for debugging
    fmt_string: String,
    // If the current expression being visited consumes itself. Used to construct
    // `best_case_captures`.
    is_consumed: bool,
    // Top-level `let __local_bindN = &expr` statements
    local_bind_decls: Vec<Stmt>,
    // Used to avoid capturing duplicated paths
    //
    // ```rust
    // let a = 1i32;
    // assert!(add(a, a) == 3);
    // ```
    paths: FxHashSet<Ident>,
    span: Span,
}

impl<'cx, 'a> Context<'cx, 'a> {
    pub(super) fn new(cx: &'cx ExtCtxt<'a>, span: Span) -> Self {
        Self {
            best_case_captures: <_>::default(),
            capture_decls: <_>::default(),
            cx,
            fmt_string: <_>::default(),
            is_consumed: true,
            local_bind_decls: <_>::default(),
            paths: <_>::default(),
            span,
        }
    }

    /// Builds the whole `assert!` expression. For example, `let elem = 1; assert!(elem == 1);` expands to:
    ///
    /// ```rust
    /// #![feature(generic_assert_internals)]
    /// let elem = 1;
    /// {
    ///   #[allow(unused_imports)]
    ///   use ::core::asserting::{TryCaptureGeneric, TryCapturePrintable};
    ///   let mut __capture0 = ::core::asserting::Capture::new();
    ///   let __local_bind0 = &elem;
    ///   if !(
    ///     *{
    ///       (&::core::asserting::Wrapper(__local_bind0)).try_capture(&mut __capture0);
    ///       __local_bind0
    ///     } == 1
    ///   ) {
    ///     panic!("Assertion failed: elem == 1\nWith captures:\n  elem = {:?}", __capture0)
    ///   }
    /// }
    /// ```
    pub(super) fn build(mut self, mut cond_expr: P<Expr>, panic_path: Path) -> P<Expr> {
        let expr_str = pprust::expr_to_string(&cond_expr);
        self.manage_cond_expr(&mut cond_expr);
        let initial_imports = self.build_initial_imports();
        let panic = self.build_panic(&expr_str, panic_path);
        let cond_expr_with_unlikely = self.build_unlikely(cond_expr);

        let Self { best_case_captures, capture_decls, cx, local_bind_decls, span, .. } = self;

        let mut assert_then_stmts = ThinVec::with_capacity(2);
        assert_then_stmts.extend(best_case_captures);
        assert_then_stmts.push(self.cx.stmt_expr(panic));
        let assert_then = self.cx.block(span, assert_then_stmts);

        let mut stmts = ThinVec::with_capacity(4);
        stmts.push(initial_imports);
        stmts.extend(capture_decls.into_iter().map(|c| c.decl));
        stmts.extend(local_bind_decls);
        stmts.push(
            cx.stmt_expr(cx.expr(span, ExprKind::If(cond_expr_with_unlikely, assert_then, None))),
        );
        cx.expr_block(cx.block(span, stmts))
    }

    /// Initial **trait** imports
    ///
    /// use ::core::asserting::{ ... };
    fn build_initial_imports(&self) -> Stmt {
        let nested_tree = |this: &Self, sym| {
            (
                UseTree {
                    prefix: this.cx.path(this.span, vec![Ident::with_dummy_span(sym)]),
                    kind: UseTreeKind::Simple(None),
                    span: this.span,
                },
                DUMMY_NODE_ID,
            )
        };
        self.cx.stmt_item(
            self.span,
            self.cx.item(
                self.span,
                Ident::empty(),
                thin_vec![self.cx.attr_nested_word(sym::allow, sym::unused_imports, self.span)],
                ItemKind::Use(UseTree {
                    prefix: self.cx.path(self.span, self.cx.std_path(&[sym::asserting])),
                    kind: UseTreeKind::Nested(thin_vec![
                        nested_tree(self, sym::TryCaptureGeneric),
                        nested_tree(self, sym::TryCapturePrintable),
                    ]),
                    span: self.span,
                }),
            ),
        )
    }

    /// Takes the conditional expression of `assert!` and then wraps it inside `unlikely`
    fn build_unlikely(&self, cond_expr: P<Expr>) -> P<Expr> {
        let unlikely_path = self.cx.std_path(&[sym::intrinsics, sym::unlikely]);
        self.cx.expr_call(
            self.span,
            self.cx.expr_path(self.cx.path(self.span, unlikely_path)),
            thin_vec![self.cx.expr(self.span, ExprKind::Unary(UnOp::Not, cond_expr))],
        )
    }

    /// The necessary custom `panic!(...)` expression.
    ///
    /// panic!(
    ///     "Assertion failed: ... \n With expansion: ...",
    ///     __capture0,
    ///     ...
    /// );
    fn build_panic(&self, expr_str: &str, panic_path: Path) -> P<Expr> {
        let escaped_expr_str = escape_to_fmt(expr_str);
        let initial = [
            TokenTree::token_alone(
                token::Literal(token::Lit {
                    kind: token::LitKind::Str,
                    symbol: Symbol::intern(&if self.fmt_string.is_empty() {
                        format!("Assertion failed: {escaped_expr_str}")
                    } else {
                        format!(
                            "Assertion failed: {escaped_expr_str}\nWith captures:\n{}",
                            &self.fmt_string
                        )
                    }),
                    suffix: None,
                }),
                self.span,
            ),
            TokenTree::token_alone(token::Comma, self.span),
        ];
        let captures = self.capture_decls.iter().flat_map(|cap| {
            [
                TokenTree::token_alone(token::Ident(cap.ident.name, false), cap.ident.span),
                TokenTree::token_alone(token::Comma, self.span),
            ]
        });
        self.cx.expr(
            self.span,
            ExprKind::MacCall(P(MacCall {
                path: panic_path,
                args: P(DelimArgs {
                    dspan: DelimSpan::from_single(self.span),
                    delim: MacDelimiter::Parenthesis,
                    tokens: initial.into_iter().chain(captures).collect::<TokenStream>(),
                }),
            })),
        )
    }

    /// Recursive function called until `cond_expr` and `fmt_str` are fully modified.
    ///
    /// See [Self::manage_initial_capture] and [Self::manage_try_capture]
    fn manage_cond_expr(&mut self, expr: &mut P<Expr>) {
        match &mut expr.kind {
            ExprKind::AddrOf(_, mutability, local_expr) => {
                self.with_is_consumed_management(
                    matches!(mutability, Mutability::Mut),
                    |this| this.manage_cond_expr(local_expr)
                );
            }
            ExprKind::Array(local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr);
                }
            }
            ExprKind::Binary(op, lhs, rhs) => {
                self.with_is_consumed_management(
                    matches!(
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
                    ),
                    |this| {
                        this.manage_cond_expr(lhs);
                        this.manage_cond_expr(rhs);
                    }
                );
            }
            ExprKind::Call(_, local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr);
                }
            }
            ExprKind::Cast(local_expr, _) => {
                self.manage_cond_expr(local_expr);
            }
            ExprKind::If(local_expr, _, _) => {
                self.manage_cond_expr(local_expr);
            }
            ExprKind::Index(prefix, suffix) => {
                self.manage_cond_expr(prefix);
                self.manage_cond_expr(suffix);
            }
            ExprKind::Let(_, local_expr, _) => {
                self.manage_cond_expr(local_expr);
            }
            ExprKind::Match(local_expr, _) => {
                self.manage_cond_expr(local_expr);
            }
            ExprKind::MethodCall(call) => {
                for arg in &mut call.args {
                    self.manage_cond_expr(arg);
                }
            }
            ExprKind::Path(_, Path { segments, .. }) if let [path_segment] = &segments[..] => {
                let path_ident = path_segment.ident;
                self.manage_initial_capture(expr, path_ident);
            }
            ExprKind::Paren(local_expr) => {
                self.manage_cond_expr(local_expr);
            }
            ExprKind::Range(prefix, suffix, _) => {
                if let Some(elem) = prefix {
                    self.manage_cond_expr(elem);
                }
                if let Some(elem) = suffix {
                    self.manage_cond_expr(elem);
                }
            }
            ExprKind::Repeat(local_expr, elem) => {
                self.manage_cond_expr(local_expr);
                self.manage_cond_expr(&mut elem.value);
            }
            ExprKind::Struct(elem) => {
                for field in &mut elem.fields {
                    self.manage_cond_expr(&mut field.expr);
                }
                if let StructRest::Base(local_expr) = &mut elem.rest {
                    self.manage_cond_expr(local_expr);
                }
            }
            ExprKind::Tup(local_exprs) => {
                for local_expr in local_exprs {
                    self.manage_cond_expr(local_expr);
                }
            }
            ExprKind::Unary(un_op, local_expr) => {
                self.with_is_consumed_management(
                    matches!(un_op, UnOp::Neg | UnOp::Not),
                    |this| this.manage_cond_expr(local_expr)
                );
            }
            // Expressions that are not worth or can not be captured.
            //
            // Full list instead of `_` to catch possible future inclusions and to
            // sync with the `rfc-2011-nicer-assert-messages/all-expr-kinds.rs` test.
            ExprKind::Assign(_, _, _)
            | ExprKind::AssignOp(_, _, _)
            | ExprKind::Async(_, _)
            | ExprKind::Await(_, _)
            | ExprKind::Block(_, _)
            | ExprKind::Break(_, _)
            | ExprKind::Closure(_)
            | ExprKind::ConstBlock(_)
            | ExprKind::Continue(_)
            | ExprKind::Err
            | ExprKind::Field(_, _)
            | ExprKind::ForLoop(_, _, _, _)
            | ExprKind::FormatArgs(_)
            | ExprKind::IncludedBytes(..)
            | ExprKind::InlineAsm(_)
            | ExprKind::Lit(_)
            | ExprKind::Loop(_, _, _)
            | ExprKind::MacCall(_)
            | ExprKind::OffsetOf(_, _)
            | ExprKind::Path(_, _)
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
    /// `fmt_str`, the formatting string used for debugging, is constructed to show possible
    /// captured variables.
    fn manage_initial_capture(&mut self, expr: &mut P<Expr>, path_ident: Ident) {
        if self.paths.contains(&path_ident) {
            return;
        } else {
            self.fmt_string.push_str("  ");
            self.fmt_string.push_str(path_ident.as_str());
            self.fmt_string.push_str(" = {:?}\n");
            let _ = self.paths.insert(path_ident);
        }
        let curr_capture_idx = self.capture_decls.len();
        let capture_string = format!("__capture{curr_capture_idx}");
        let ident = Ident::new(Symbol::intern(&capture_string), self.span);
        let init_std_path = self.cx.std_path(&[sym::asserting, sym::Capture, sym::new]);
        let init = self.cx.expr_call(
            self.span,
            self.cx.expr_path(self.cx.path(self.span, init_std_path)),
            ThinVec::new(),
        );
        let capture = Capture { decl: self.cx.stmt_let(self.span, true, ident, init), ident };
        self.capture_decls.push(capture);
        self.manage_try_capture(ident, curr_capture_idx, expr);
    }

    /// Tries to copy `__local_bindN` into `__captureN`.
    ///
    /// *{
    ///    (&Wrapper(__local_bindN)).try_capture(&mut __captureN);
    ///    __local_bindN
    /// }
    fn manage_try_capture(&mut self, capture: Ident, curr_capture_idx: usize, expr: &mut P<Expr>) {
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
            thin_vec![self.cx.expr_path(Path::from_ident(local_bind))],
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
                expr_paren(self.cx, self.span, self.cx.expr_addr_of(self.span, wrapper)),
                thin_vec![expr_addr_of_mut(
                    self.cx,
                    self.span,
                    self.cx.expr_path(Path::from_ident(capture)),
                )],
                self.span,
            ))
            .add_trailing_semicolon();
        let local_bind_path = self.cx.expr_path(Path::from_ident(local_bind));
        let rslt = if self.is_consumed {
            let ret = self.cx.stmt_expr(local_bind_path);
            self.cx.expr_block(self.cx.block(self.span, thin_vec![try_capture_call, ret]))
        } else {
            self.best_case_captures.push(try_capture_call);
            local_bind_path
        };
        *expr = self.cx.expr_deref(self.span, rslt);
    }

    // Calls `f` with the internal `is_consumed` set to `curr_is_consumed` and then
    // sets the internal `is_consumed` back to its original value.
    fn with_is_consumed_management(&mut self, curr_is_consumed: bool, f: impl FnOnce(&mut Self)) {
        let prev_is_consumed = self.is_consumed;
        self.is_consumed = curr_is_consumed;
        f(self);
        self.is_consumed = prev_is_consumed;
    }
}

/// Information about a captured element.
#[derive(Debug)]
struct Capture {
    // Generated indexed `Capture` statement.
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
    seg: PathSegment,
    receiver: P<Expr>,
    args: ThinVec<P<Expr>>,
    span: Span,
) -> P<Expr> {
    cx.expr(span, ExprKind::MethodCall(Box::new(MethodCall { seg, receiver, args, span })))
}

fn expr_paren(cx: &ExtCtxt<'_>, sp: Span, e: P<Expr>) -> P<Expr> {
    cx.expr(sp, ExprKind::Paren(e))
}
