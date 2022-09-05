use super::server::RpcContext;
use super::*;

use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct TokenStream {
    pub(crate) tokens: Rc<Vec<crate::TokenTree>>,
}

impl TokenStream {
    pub(crate) fn new(tokens: Vec<crate::TokenTree>) -> Self {
        TokenStream { tokens: Rc::new(tokens) }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl<S> Encode<S> for TokenStream {
    fn encode(self, w: &mut Writer, s: &mut S) {
        let tts: Vec<_> = self
            .tokens
            .iter()
            .map(|tt| match tt {
                crate::TokenTree::Group(group) => TokenTree::Group(group.0.clone()),
                crate::TokenTree::Punct(punct) => TokenTree::Punct(punct.0.clone()),
                crate::TokenTree::Ident(ident) => TokenTree::Ident(ident.0.clone()),
                crate::TokenTree::Literal(literal) => TokenTree::Literal(literal.0.clone()),
            })
            .collect();
        tts.encode(w, s)
    }
}

impl<S: server::Server> DecodeMut<'_, '_, client::HandleStore<server::MarkedTypes<S>>>
    for Marked<S::TokenStream, TokenStream>
{
    fn decode(r: &mut Reader<'_>, s: &mut client::HandleStore<server::MarkedTypes<S>>) -> Self {
        let tts: Vec<_> = DecodeMut::decode(r, s);
        s.rpc_context.tokenstream_from_tts(tts.into_iter())
    }
}

impl<S: server::Server> Encode<client::HandleStore<server::MarkedTypes<S>>>
    for Marked<S::TokenStream, TokenStream>
{
    fn encode(self, w: &mut Writer, s: &mut client::HandleStore<server::MarkedTypes<S>>) {
        let tts = s.rpc_context.tts_from_tokenstream(self);
        tts.encode(w, s);
    }
}

impl<S> DecodeMut<'_, '_, S> for TokenStream {
    fn decode(r: &mut Reader<'_>, s: &mut S) -> Self {
        TokenStream::new(
            <Vec<_>>::decode(r, s)
                .into_iter()
                .map(|tt| match tt {
                    TokenTree::Group(group) => crate::TokenTree::Group(crate::Group(group)),
                    TokenTree::Punct(punct) => crate::TokenTree::Punct(crate::Punct(punct)),
                    TokenTree::Ident(ident) => crate::TokenTree::Ident(crate::Ident(ident)),
                    TokenTree::Literal(literal) => {
                        crate::TokenTree::Literal(crate::Literal(literal))
                    }
                })
                .collect(),
        )
    }
}
