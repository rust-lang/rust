use rustc_data_structures::thin_vec::ThinVec;

use syntax::ast;
use syntax::ext::base::{self, *};
use syntax::parse::token::{self, Token};
use syntax::ptr::P;
use syntax_pos::Span;
use syntax_pos::symbol::Symbol;
use syntax::tokenstream::{TokenTree, TokenStream};

pub fn expand_concat_idents<'cx>(cx: &'cx mut ExtCtxt<'_>,
                                 sp: Span,
                                 tts: TokenStream)
                                 -> Box<dyn base::MacResult + 'cx> {
    if tts.is_empty() {
        cx.span_err(sp, "concat_idents! takes 1 or more arguments.");
        return DummyResult::any(sp);
    }

    let mut res_str = String::new();
    for (i, e) in tts.into_trees().enumerate() {
        if i & 1 == 1 {
            match e {
                TokenTree::Token(Token { kind: token::Comma, .. }) => {}
                _ => {
                    cx.span_err(sp, "concat_idents! expecting comma.");
                    return DummyResult::any(sp);
                }
            }
        } else {
            match e {
                TokenTree::Token(Token { kind: token::Ident(name, _), .. }) =>
                    res_str.push_str(&name.as_str()),
                _ => {
                    cx.span_err(sp, "concat_idents! requires ident args.");
                    return DummyResult::any(sp);
                }
            }
        }
    }

    let ident = ast::Ident::new(Symbol::intern(&res_str), cx.with_legacy_ctxt(sp));

    struct ConcatIdentsResult { ident: ast::Ident }

    impl base::MacResult for ConcatIdentsResult {
        fn make_expr(self: Box<Self>) -> Option<P<ast::Expr>> {
            Some(P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprKind::Path(None, ast::Path::from_ident(self.ident)),
                span: self.ident.span,
                attrs: ThinVec::new(),
            }))
        }

        fn make_ty(self: Box<Self>) -> Option<P<ast::Ty>> {
            Some(P(ast::Ty {
                id: ast::DUMMY_NODE_ID,
                node: ast::TyKind::Path(None, ast::Path::from_ident(self.ident)),
                span: self.ident.span,
            }))
        }
    }

    Box::new(ConcatIdentsResult { ident })
}
