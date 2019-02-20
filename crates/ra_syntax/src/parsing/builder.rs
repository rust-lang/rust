use crate::{
    parsing::parser_impl::Sink,
    syntax_node::{GreenNode, RaTypes},
    SmolStr, SyntaxKind, SyntaxError,
};

use rowan::GreenNodeBuilder;

pub(crate) struct GreenBuilder {
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl GreenBuilder {
    pub(crate) fn new() -> GreenBuilder {
        GreenBuilder { errors: Vec::new(), inner: GreenNodeBuilder::new() }
    }
}

impl Sink for GreenBuilder {
    type Tree = (GreenNode, Vec<SyntaxError>);

    fn leaf(&mut self, kind: SyntaxKind, text: SmolStr) {
        self.inner.leaf(kind, text);
    }

    fn start_branch(&mut self, kind: SyntaxKind) {
        self.inner.start_internal(kind)
    }

    fn finish_branch(&mut self) {
        self.inner.finish_internal();
    }

    fn error(&mut self, error: SyntaxError) {
        self.errors.push(error)
    }

    fn finish(self) -> (GreenNode, Vec<SyntaxError>) {
        (self.inner.finish(), self.errors)
    }
}
