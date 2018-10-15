use rowan::GreenNodeBuilder;
use crate::{
    TextUnit, SmolStr,
    parser_impl::Sink,
    yellow::{GreenNode, SyntaxError, RaTypes},
    SyntaxKind,
};

pub(crate) struct GreenBuilder {
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl GreenBuilder {
    pub(crate) fn new() -> GreenBuilder {
        GreenBuilder {
            errors: Vec::new(),
            inner: GreenNodeBuilder::new(),
        }
    }
}

impl Sink for GreenBuilder {
    type Tree = (GreenNode, Vec<SyntaxError>);

    fn leaf(&mut self, kind: SyntaxKind, text: SmolStr) {
        self.inner.leaf(kind, text);
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        self.inner.start_internal(kind)
    }

    fn finish_internal(&mut self) {
        self.inner.finish_internal();
    }

    fn error(&mut self, message: String, offset: TextUnit) {
        let error = SyntaxError { msg: message, offset };
        self.errors.push(error)
    }

    fn finish(self) -> (GreenNode, Vec<SyntaxError>) {
        (self.inner.finish(), self.errors)
    }
}
