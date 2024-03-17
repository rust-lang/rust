mod context;

use crate::edition_panic::use_panic_2021;
use crate::errors;
use rustc_ast::ptr::P;
use rustc_ast::token::Delimiter;
use rustc_ast::tokenstream::{DelimSpan, TokenStream};
use rustc_ast::{self as ast, token};
use rustc_ast::{DelimArgs, Expr, ExprKind, MacCall, Path, PathSegment, UnOp};
use rustc_ast_pretty::pprust;
use rustc_errors::PResult;
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacEager, MacroExpanderResult};
use rustc_parse::parser::Parser;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use thin_vec::thin_vec;

pub fn expand_assert<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let Assert { cond_expr, inner_cond_expr, custom_message } = match parse_assert(cx, span, tts) {
        Ok(assert) => assert,
        Err(err) => {
            let guar = err.emit();
            return ExpandResult::Ready(DummyResult::any(span, guar));
        }
    };

    // `core::panic` and `std::panic` are different macros, so we use call-site
    // context to pick up whichever is currently in scope.
    let call_site_span = cx.with_call_site_ctxt(span);

    let panic_path = || {
        if use_panic_2021(span) {
            // On edition 2021, we always call `$crate::panic::panic_2021!()`.
            Path {
                span: call_site_span,
                segments: cx
                    .std_path(&[sym::panic, sym::panic_2021])
                    .into_iter()
                    .map(|ident| PathSegment::from_ident(ident))
                    .collect(),
                tokens: None,
            }
        } else {
            // Before edition 2021, we call `panic!()` unqualified,
            // such that it calls either `std::panic!()` or `core::panic!()`.
            Path::from_ident(Ident::new(sym::panic, call_site_span))
        }
    };

    // Simply uses the user provided message instead of generating custom outputs
    let expr = if let Some(tokens) = custom_message {
        let then = cx.expr(
            call_site_span,
            ExprKind::MacCall(P(MacCall {
                path: panic_path(),
                args: P(DelimArgs {
                    dspan: DelimSpan::from_single(call_site_span),
                    delim: Delimiter::Parenthesis,
                    tokens,
                }),
            })),
        );
        expr_if_not(cx, call_site_span, cond_expr, then, None)
    }
    // If `generic_assert` is enabled, generates rich captured outputs
    //
    // FIXME(c410-f3r) See https://github.com/rust-lang/rust/issues/96949
    else if cx.ecfg.features.generic_assert {
        // FIXME(estebank): we use the condition the user passed without coercing to `bool` when
        // `generic_assert` is enabled, but we could use `cond_expr` instead.
        context::Context::new(cx, call_site_span).build(inner_cond_expr, panic_path())
    }
    // If `generic_assert` is not enabled, only outputs a literal "assertion failed: ..."
    // string
    else {
        // Pass our own message directly to $crate::panicking::panic(),
        // because it might contain `{` and `}` that should always be
        // passed literally.
        let then = cx.expr_call_global(
            call_site_span,
            cx.std_path(&[sym::panicking, sym::panic]),
            thin_vec![cx.expr_str(
                DUMMY_SP,
                Symbol::intern(&format!(
                    "assertion failed: {}",
                    pprust::expr_to_string(&inner_cond_expr)
                )),
            )],
        );
        expr_if_not(cx, call_site_span, cond_expr, then, None)
    };

    ExpandResult::Ready(MacEager::expr(expr))
}

// `assert!($cond_expr, $custom_message)`
struct Assert {
    // `{ let assert_macro: bool = $cond_expr; assert_macro }`
    cond_expr: P<Expr>,
    // We keep the condition without the `bool` coercion for the panic message.
    inner_cond_expr: P<Expr>,
    custom_message: Option<TokenStream>,
}

// if !{ ... } { ... } else { ... }
fn expr_if_not(
    cx: &ExtCtxt<'_>,
    span: Span,
    cond: P<Expr>,
    then: P<Expr>,
    els: Option<P<Expr>>,
) -> P<Expr> {
    cx.expr_if(span, cx.expr(span, ExprKind::Unary(UnOp::Not, cond)), then, els)
}

fn parse_assert<'a>(cx: &mut ExtCtxt<'a>, sp: Span, stream: TokenStream) -> PResult<'a, Assert> {
    let mut parser = cx.new_parser_from_tts(stream);

    if parser.token == token::Eof {
        return Err(cx.dcx().create_err(errors::AssertRequiresBoolean { span: sp }));
    }

    let inner_cond_expr = parser.parse_expr()?;

    // Some crates use the `assert!` macro in the following form (note extra semicolon):
    //
    // assert!(
    //     my_function();
    // );
    //
    // Emit an error about semicolon and suggest removing it.
    if parser.token == token::Semi {
        cx.dcx().emit_err(errors::AssertRequiresExpression { span: sp, token: parser.token.span });
        parser.bump();
    }

    // Some crates use the `assert!` macro in the following form (note missing comma before
    // message):
    //
    // assert!(true "error message");
    //
    // Emit an error and suggest inserting a comma.
    let custom_message =
        if let token::Literal(token::Lit { kind: token::Str, .. }) = parser.token.kind {
            let comma = parser.prev_token.span.shrink_to_hi();
            cx.dcx().emit_err(errors::AssertMissingComma { span: parser.token.span, comma });

            parse_custom_message(&mut parser)
        } else if parser.eat(&token::Comma) {
            parse_custom_message(&mut parser)
        } else {
            None
        };

    if parser.token != token::Eof {
        return parser.unexpected();
    }

    let cond_expr = expand_cond(cx, parser, inner_cond_expr.clone());
    Ok(Assert { cond_expr, inner_cond_expr, custom_message })
}

fn expand_cond(cx: &ExtCtxt<'_>, parser: Parser<'_>, cond_expr: P<Expr>) -> P<Expr> {
    let span = cx.with_call_site_ctxt(cond_expr.span);
    // Coerce the expression to `bool` for more accurate errors. If `assert!` is passed an
    // expression that isn't `bool`, the type error will point at only the expression and not the
    // entire macro call. If a non-`bool` is passed that doesn't implement `trait Not`, we won't
    // talk about traits, we'll just state the appropriate type error.
    // `let assert_macro: bool = $expr;`
    let ident = Ident::new(sym::assert_macro, span);
    let local = P(ast::Local {
        ty: Some(P(ast::Ty {
            kind: ast::TyKind::Path(None, ast::Path::from_ident(Ident::new(sym::bool, span))),
            id: ast::DUMMY_NODE_ID,
            span,
            tokens: None,
        })),
        pat: parser.mk_pat_ident(span, ast::BindingAnnotation::NONE, ident),
        kind: ast::LocalKind::Init(cond_expr),
        id: ast::DUMMY_NODE_ID,
        span,
        colon_sp: None,
        attrs: Default::default(),
        tokens: None,
    });
    // `{ let assert_macro: bool = $expr; assert_macro }`
    parser.mk_expr(
        span,
        ast::ExprKind::Block(
            parser.mk_block(
                thin_vec![
                    parser.mk_stmt(span, ast::StmtKind::Let(local)),
                    parser.mk_stmt(
                        span,
                        ast::StmtKind::Expr(parser.mk_expr(
                            span,
                            ast::ExprKind::Path(None, ast::Path::from_ident(ident))
                        )),
                    ),
                ],
                ast::BlockCheckMode::Default,
                span,
            ),
            None,
        ),
    )
}

fn parse_custom_message(parser: &mut Parser<'_>) -> Option<TokenStream> {
    let ts = parser.parse_tokens();
    if !ts.is_empty() { Some(ts) } else { None }
}
