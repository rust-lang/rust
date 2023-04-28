use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::PResult;
use rustc_expand::base::{self, *};
use rustc_macros::Diagnostic;
use rustc_parse::parser::Parser;
use rustc_span::{symbol::Ident, Span};

#[derive(Diagnostic)]
#[diag(builtin_macros_offset_of_expected_field)]
struct ExpectedField {
    #[primary_span]
    span: Span,
}

#[derive(Diagnostic)]
#[diag(builtin_macros_offset_of_expected_two_args)]
struct ExpectedTwoArgs {
    #[primary_span]
    span: Span,
}

fn parse_field<'a>(cx: &ExtCtxt<'a>, p: &mut Parser<'a>) -> PResult<'a, Ident> {
    let token = p.token.uninterpolate();
    let field = match token.kind {
        token::Ident(name, _) => Ident::new(name, token.span),
        token::Literal(token::Lit { kind: token::Integer, symbol, suffix: None }) => {
            Ident::new(symbol, token.span)
        }
        _ => return Err(cx.create_err(ExpectedField { span: p.token.span })),
    };

    p.bump();

    Ok(field)
}

fn parse_args<'a>(
    cx: &mut ExtCtxt<'a>,
    sp: Span,
    tts: TokenStream,
) -> PResult<'a, (P<ast::Ty>, P<[Ident]>)> {
    let mut p = cx.new_parser_from_tts(tts);

    let container = p.parse_ty()?;

    p.expect(&token::Comma)?;

    if p.eat(&token::Eof) {
        return Err(cx.create_err(ExpectedTwoArgs { span: sp }));
    }

    let mut fields = Vec::new();

    loop {
        let field = parse_field(cx, &mut p)?;
        fields.push(field);

        if p.eat(&token::Dot) {
            continue;
        }

        p.eat(&token::Comma);

        if !p.eat(&token::Eof) {
            return Err(cx.create_err(ExpectedTwoArgs { span: sp }));
        }

        break;
    }

    Ok((container, fields.into()))
}

pub fn expand_offset_of<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    match parse_args(cx, sp, tts) {
        Ok((container, fields)) => {
            let expr = P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                kind: ast::ExprKind::OffsetOf(container, fields),
                span: sp,
                attrs: ast::AttrVec::new(),
                tokens: None,
            });

            MacEager::expr(expr)
        }
        Err(mut err) => {
            err.emit();
            DummyResult::any(sp)
        }
    }
}
