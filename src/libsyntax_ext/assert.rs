use errors::{Applicability, DiagnosticBuilder};

use syntax::ast::{self, *};
use syntax::source_map::Spanned;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::parse::token::{self, TokenKind};
use syntax::parse::parser::Parser;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::symbol::{sym, Symbol};
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax_pos::{Span, DUMMY_SP};

pub fn expand_assert<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: &[TokenTree],
) -> Box<dyn MacResult + 'cx> {
    let Assert { cond_expr, custom_message } = match parse_assert(cx, sp, tts) {
        Ok(assert) => assert,
        Err(mut err) => {
            err.emit();
            return DummyResult::expr(sp);
        }
    };

    let sp = sp.apply_mark(cx.current_expansion.mark);
    let panic_call = Mac_ {
        path: Path::from_ident(Ident::new(sym::panic, sp)),
        tts: custom_message.unwrap_or_else(|| {
            TokenStream::from(TokenTree::token(
                TokenKind::lit(token::Str, Symbol::intern(&format!(
                    "assertion failed: {}",
                    pprust::expr_to_string(&cond_expr).escape_debug()
                )), None),
                DUMMY_SP,
            ))
        }).into(),
        delim: MacDelimiter::Parenthesis,
    };
    let if_expr = cx.expr_if(
        sp,
        cx.expr(sp, ExprKind::Unary(UnOp::Not, cond_expr)),
        cx.expr(
            sp,
            ExprKind::Mac(Spanned {
                span: sp,
                node: panic_call,
            }),
        ),
        None,
    );
    MacEager::expr(if_expr)
}

struct Assert {
    cond_expr: P<ast::Expr>,
    custom_message: Option<TokenStream>,
}

fn parse_assert<'a>(
    cx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: &[TokenTree]
) -> Result<Assert, DiagnosticBuilder<'a>> {
    let mut parser = cx.new_parser_from_tts(tts);

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
    // Warn about semicolon and suggest removing it. Eventually, this should be turned into an
    // error.
    if parser.token == token::Semi {
        let mut err = cx.struct_span_warn(sp, "macro requires an expression as an argument");
        err.span_suggestion(
            parser.token.span,
            "try removing semicolon",
            String::new(),
            Applicability::MaybeIncorrect
        );
        err.note("this is going to be an error in the future");
        err.emit();

        parser.bump();
    }

    // Some crates use the `assert!` macro in the following form (note missing comma before
    // message):
    //
    // assert!(true "error message");
    //
    // Parse this as an actual message, and suggest inserting a comma. Eventually, this should be
    // turned into an error.
    let custom_message = if let token::Literal(token::Lit { kind: token::Str, .. })
                                = parser.token.kind {
        let mut err = cx.struct_span_warn(parser.token.span, "unexpected string literal");
        let comma_span = cx.source_map().next_point(parser.prev_span);
        err.span_suggestion_short(
            comma_span,
            "try adding a comma",
            ", ".to_string(),
            Applicability::MaybeIncorrect
        );
        err.note("this is going to be an error in the future");
        err.emit();

        parse_custom_message(&mut parser)
    } else if parser.eat(&token::Comma) {
        parse_custom_message(&mut parser)
    } else {
        None
    };

    if parser.token != token::Eof {
        parser.expect_one_of(&[], &[])?;
        unreachable!();
    }

    Ok(Assert { cond_expr, custom_message })
}

fn parse_custom_message(parser: &mut Parser<'_>) -> Option<TokenStream> {
    let ts = parser.parse_tokens();
    if !ts.is_empty() {
        Some(ts)
    } else {
        None
    }
}
