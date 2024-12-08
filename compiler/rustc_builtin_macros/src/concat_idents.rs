use rustc_ast::ptr::P;
use rustc_ast::token::{self, Token};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::{AttrVec, DUMMY_NODE_ID, Expr, ExprKind, Path, Ty, TyKind};
use rustc_expand::base::{DummyResult, ExpandResult, ExtCtxt, MacResult, MacroExpanderResult};
use rustc_span::Span;
use rustc_span::symbol::{Ident, Symbol};

use crate::errors;

pub(crate) fn expand_concat_idents<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    if tts.is_empty() {
        let guar = cx.dcx().emit_err(errors::ConcatIdentsMissingArgs { span: sp });
        return ExpandResult::Ready(DummyResult::any(sp, guar));
    }

    let mut res_str = String::new();
    for (i, e) in tts.trees().enumerate() {
        if i & 1 == 1 {
            match e {
                TokenTree::Token(Token { kind: token::Comma, .. }, _) => {}
                _ => {
                    let guar = cx.dcx().emit_err(errors::ConcatIdentsMissingComma { span: sp });
                    return ExpandResult::Ready(DummyResult::any(sp, guar));
                }
            }
        } else {
            if let TokenTree::Token(token, _) = e {
                if let Some((ident, _)) = token.ident() {
                    res_str.push_str(ident.name.as_str());
                    continue;
                }
            }

            let guar = cx.dcx().emit_err(errors::ConcatIdentsIdentArgs { span: sp });
            return ExpandResult::Ready(DummyResult::any(sp, guar));
        }
    }

    let ident = Ident::new(Symbol::intern(&res_str), cx.with_call_site_ctxt(sp));

    struct ConcatIdentsResult {
        ident: Ident,
    }

    impl MacResult for ConcatIdentsResult {
        fn make_expr(self: Box<Self>) -> Option<P<Expr>> {
            Some(P(Expr {
                id: DUMMY_NODE_ID,
                kind: ExprKind::Path(None, Path::from_ident(self.ident)),
                span: self.ident.span,
                attrs: AttrVec::new(),
                tokens: None,
            }))
        }

        fn make_ty(self: Box<Self>) -> Option<P<Ty>> {
            Some(P(Ty {
                id: DUMMY_NODE_ID,
                kind: TyKind::Path(None, Path::from_ident(self.ident)),
                span: self.ident.span,
                tokens: None,
            }))
        }
    }

    ExpandResult::Ready(Box::new(ConcatIdentsResult { ident }))
}
