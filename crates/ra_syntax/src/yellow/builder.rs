use rowan::GreenNodeBuilder;
use {
    parser_impl::Sink,
    yellow::{GreenNode, SyntaxError, RaTypes},
    SyntaxKind, TextRange, TextUnit,
};

pub(crate) struct GreenBuilder<'a> {
    text: &'a str,
    pos: TextUnit,
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl<'a> Sink<'a> for GreenBuilder<'a> {
    type Tree = (GreenNode, Vec<SyntaxError>);

    fn new(text: &'a str) -> Self {
        GreenBuilder {
            text,
            pos: 0.into(),
            errors: Vec::new(),
            inner: GreenNodeBuilder::new(),
        }
    }

    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let range = TextRange::offset_len(self.pos, len);
        self.pos += len;
        let text = self.text[range].into();
        self.inner.leaf(kind, text);
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        self.inner.start_internal(kind)
    }

    fn finish_internal(&mut self) {
        self.inner.finish_internal();
    }

    fn error(&mut self, message: String) {
        self.errors.push(SyntaxError {
            msg: message,
            offset: self.pos,
        })
    }

    fn finish(self) -> (GreenNode, Vec<SyntaxError>) {
        (self.inner.finish(), self.errors)
    }
}
