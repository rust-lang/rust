use crate::{
    SmolStr, SyntaxKind, SyntaxError, SyntaxErrorKind, TextUnit,
    parsing::{TreeSink, ParseError},
    syntax_node::{GreenNode, RaTypes},
};

use rowan::GreenNodeBuilder;

pub(crate) struct GreenBuilder {
    text_pos: TextUnit,
    errors: Vec<SyntaxError>,
    inner: GreenNodeBuilder<RaTypes>,
}

impl Default for GreenBuilder {
    fn default() -> GreenBuilder {
        GreenBuilder {
            text_pos: TextUnit::default(),
            errors: Vec::new(),
            inner: GreenNodeBuilder::new(),
        }
    }
}

impl TreeSink for GreenBuilder {
    type Tree = (GreenNode, Vec<SyntaxError>);

    fn leaf(&mut self, kind: SyntaxKind, text: SmolStr) {
        self.text_pos += TextUnit::of_str(text.as_str());
        self.inner.leaf(kind, text);
    }

    fn start_branch(&mut self, kind: SyntaxKind) {
        self.inner.start_internal(kind)
    }

    fn finish_branch(&mut self) {
        self.inner.finish_internal();
    }

    fn error(&mut self, error: ParseError) {
        let error = SyntaxError::new(SyntaxErrorKind::ParseError(error), self.text_pos);
        self.errors.push(error)
    }

    fn finish(self) -> (GreenNode, Vec<SyntaxError>) {
        (self.inner.finish(), self.errors)
    }
}
