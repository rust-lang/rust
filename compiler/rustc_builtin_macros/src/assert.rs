use crate::edition_panic::use_panic_2021;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::{DelimSpan, TokenStream};
use rustc_ast::{self as ast, *};
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_expand::base::*;
use rustc_parse::parser::Parser;
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};

pub fn expand_assert<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    let Assert { cond_expr, custom_message } = match parse_assert(cx, span, tts) {
        Ok(assert) => assert,
        Err(mut err) => {
            err.emit();
            return DummyResult::any(span);
        }
    };

    // `core::panic` and `std::panic` are different macros, so we use call-site
    // context to pick up whichever is currently in scope.
    let sp = cx.with_call_site_ctxt(span);

    let panic_call = if let Some(tokens) = custom_message {
        let path = if use_panic_2021(span) {
            // On edition 2021, we always call `$crate::panic::panic_2021!()`.
            Path {
                span: sp,
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
            Path::from_ident(Ident::new(sym::panic, sp))
        };
        // Pass the custom message to panic!().
        cx.expr(
            sp,
            ExprKind::MacCall(MacCall {
                path,
                args: P(MacArgs::Delimited(
                    DelimSpan::from_single(sp),
                    MacDelimiter::Parenthesis,
                    tokens,
                )),
                prior_type_ascription: None,
            }),
        )
    } else {
        // Pass our own message directly to $crate::panicking::panic(),
        // because it might contain `{` and `}` that should always be
        // passed literally.
        cx.expr_call_global(
            sp,
            cx.std_path(&[sym::panicking, sym::panic]),
            vec![cx.expr_str(
                DUMMY_SP,
                Symbol::intern(&format!(
                    "assertion failed: {}",
                    pprust::expr_to_string(&cond_expr).escape_debug()
                )),
            )],
        )
    };
    let if_expr =
        cx.expr_if(sp, cx.expr(sp, ExprKind::Unary(UnOp::Not, cond_expr)), panic_call, None);
    MacEager::expr(if_expr)
}

struct Assert {
    cond_expr: P<ast::Expr>,
    custom_message: Option<TokenStream>,
}

fn parse_assert<'a>(
    cx: &mut ExtCtxt<'a>,
    sp: Span,
    stream: TokenStream,
) -> Result<Assert, DiagnosticBuilder<'a>> {
    let mut parser = cx.new_parser_from_tts(stream);

    if parser.token == token::Eof {
        let mut err = cx.struct_span_err(sp, "macro requires a boolean expression as an argument");
        err.span_label(sp, "boolean expression required");
        return Err(err);
    }

    let cond_expr = parser.parse_expr()?;

    // Some crates use the `assert!` macro in the following form (note extra semicolon):
    //
    // assert!(
    //     my_function();
    // );
    //
    // Emit an error about semicolon and suggest removing it.
    if parser.token == token::Semi {
        let mut err = cx.struct_span_err(sp, "macro requires an expression as an argument");
        err.span_suggestion(
            parser.token.span,
            "try removing semicolon",
            String::new(),
            Applicability::MaybeIncorrect,
        );
        err.emit();

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
            let mut err = cx.struct_span_err(parser.token.span, "unexpected string literal");
            let comma_span = parser.prev_token.span.shrink_to_hi();
            err.span_suggestion_short(
                comma_span,
                "try adding a comma",
                ", ".to_string(),
                Applicability::MaybeIncorrect,
            );
            err.emit();

            parse_custom_message(&mut parser)
        } else if parser.eat(&token::Comma) {
            parse_custom_message(&mut parser)
        } else {
            None
        };

    if parser.token != token::Eof {
        return parser.unexpected();
    }

    Ok(Assert { cond_expr, custom_message })
}

fn parse_custom_message(parser: &mut Parser<'_>) -> Option<TokenStream> {
    let ts = parser.parse_tokens();
    if !ts.is_empty() { Some(ts) } else { None }
}
