use rustc_ast::tokenstream::TokenStream;
use rustc_ast::NodeId;
use rustc_macros::HashStable_Generic;
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::{ErrorGuaranteed, LocalExpnId, Span};

pub trait TcxMacroExpander {
    fn expand(
        &self,
        _sess: &Session,
        _span: Span,
        _input: TokenStream,
        _expand_id: LocalExpnId,
    ) -> Result<(TokenStream, usize), CanRetry>;

    fn name(&self) -> Ident;

    fn arm_span(&self, rhs: usize) -> Span;

    fn node_id(&self) -> NodeId;
}

#[derive(Copy, Clone, HashStable_Generic, Debug)]
pub enum CanRetry {
    Yes,
    /// We are not allowed to retry macro expansion as a fatal error has been emitted already.
    No(ErrorGuaranteed),
}
