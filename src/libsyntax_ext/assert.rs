use syntax::ast::*;
use syntax::source_map::Spanned;
use syntax::ext::base::*;
use syntax::ext::build::AstBuilder;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::symbol::Symbol;
use syntax::tokenstream::{TokenStream, TokenTree};
use syntax_pos::{Span, DUMMY_SP};

pub fn expand_assert<'cx>(
    cx: &'cx mut ExtCtxt,
    sp: Span,
    tts: &[TokenTree],
) -> Box<dyn MacResult + 'cx> {
    let mut parser = cx.new_parser_from_tts(tts);

    if parser.token == token::Eof {
        cx.struct_span_err(sp, "macro requires a boolean expression as an argument")
            .span_label(sp, "boolean expression required")
            .emit();
        return DummyResult::expr(sp);
    }

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

    let sp = sp.apply_mark(cx.current_expansion.mark);
    let panic_call = Mac_ {
        path: Path::from_ident(Ident::new(Symbol::intern("panic"), sp)),
        tts: if let Some(ts) = custom_msg_args {
            ts.into()
        } else {
            TokenStream::from(TokenTree::Token(
                DUMMY_SP,
                token::Literal(
                    token::Lit::Str_(Name::intern(&format!(
                        "assertion failed: {}",
                        pprust::expr_to_string(&cond_expr).escape_debug()
                    ))),
                    None,
                ),
            )).into()
        },
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
