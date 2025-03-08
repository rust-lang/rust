use rustc_ast::ptr::P;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{
    CaptureBy, ClosureBinder, Const, CoroutineKind, DUMMY_NODE_ID, Expr, ExprKind, ast, token,
};
use rustc_errors::PResult;
use rustc_expand::base::{self, DummyResult, ExpandResult, ExtCtxt, MacroExpanderResult};
use rustc_span::Span;

pub(crate) fn expand<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let closure = match parse_closure(cx, sp, tts) {
        Ok(parsed) => parsed,
        Err(err) => {
            return ExpandResult::Ready(DummyResult::any(sp, err.emit()));
        }
    };

    ExpandResult::Ready(base::MacEager::expr(closure))
}

fn parse_closure<'a>(
    cx: &mut ExtCtxt<'a>,
    span: Span,
    stream: TokenStream,
) -> PResult<'a, P<Expr>> {
    let mut parser = cx.new_parser_from_tts(stream);
    let mut closure_parser = parser.clone();

    let coroutine_kind = Some(CoroutineKind::Gen {
        span,
        closure_id: DUMMY_NODE_ID,
        return_impl_trait_id: DUMMY_NODE_ID,
    });

    match closure_parser.parse_expr() {
        Ok(mut closure) => {
            if let ast::ExprKind::Closure(c) = &mut closure.kind {
                if let Some(kind) = c.coroutine_kind {
                    cx.dcx().span_err(kind.span(), "only plain closures allowed in `iter!`");
                }
                c.coroutine_kind = coroutine_kind;
                if closure_parser.token != token::Eof {
                    closure_parser.unexpected()?;
                }
                return Ok(closure);
            }
        }
        Err(diag) => diag.cancel(),
    }

    let lo = parser.token.span.shrink_to_lo();
    let block = parser.parse_block_tail(
        lo,
        ast::BlockCheckMode::Default,
        rustc_parse::parser::AttemptLocalParseRecovery::No,
    )?;
    let fn_decl = cx.fn_decl(Default::default(), ast::FnRetTy::Default(span));
    let closure = ast::Closure {
        binder: ClosureBinder::NotPresent,
        capture_clause: CaptureBy::Ref,
        constness: Const::No,
        coroutine_kind,
        movability: ast::Movability::Movable,
        fn_decl,
        body: cx.expr_block(block),
        fn_decl_span: span,
        fn_arg_span: span,
    };
    if parser.token != token::Eof {
        parser.unexpected()?;
    }
    let span = lo.to(parser.token.span);
    Ok(cx.expr(span, ExprKind::Closure(Box::new(closure))))
}
