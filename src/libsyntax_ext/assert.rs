use crate::errors::DiagnosticBuilder;

use syntax::ast::{self, *};
use syntax::source_map::Spanned;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::symbol::Symbol;
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
        path: Path::from_ident(Ident::new(Symbol::intern("panic"), sp)),
        tts: custom_message.unwrap_or_else(|| {
            TokenStream::from(TokenTree::Token(
                DUMMY_SP,
                token::Literal(
                    token::Lit::Str_(Name::intern(&format!(
                        "assertion failed: {}",
                        pprust::expr_to_string(&cond_expr).escape_debug()
                    ))),
                    None,
                ),
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

    Ok(Assert {
        cond_expr: parser.parse_expr()?,
        custom_message: if parser.eat(&token::Comma) {
            let ts = parser.parse_tokens();
            if !ts.is_empty() {
                Some(ts)
            } else {
                None
            }
        } else {
            None
        },
    })
}
