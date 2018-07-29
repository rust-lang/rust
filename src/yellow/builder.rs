use std::sync::Arc;
use {
    SyntaxKind, TextRange, TextUnit,
    yellow::{SyntaxNode, GreenNode, SyntaxError},
    parser::Sink
};

pub(crate) struct GreenBuilder {
    text: String,
    stack: Vec<GreenNode>,
    pos: TextUnit,
    root: Option<GreenNode>,
    errors: Vec<SyntaxError>,
}

impl GreenBuilder {

}

impl Sink for GreenBuilder {
    type Tree = SyntaxNode;

    fn new(text: String) -> Self {
        GreenBuilder {
            text,
            stack: Vec::new(),
            pos: 0.into(),
            root: None,
            errors: Vec::new(),
        }
    }

    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let range = TextRange::offset_len(self.pos, len);
        self.pos += len;
        let text = self.text[range].to_owned();
        let parent = self.stack.last_mut().unwrap();
        if kind.is_trivia() {
            parent.push_trivia(kind, text);
        } else {
            let node = GreenNode::new_leaf(kind, text);
            parent.push_child(Arc::new(node));
        }
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        self.stack.push(GreenNode::new_branch(kind))
    }

    fn finish_internal(&mut self) {
        let node = self.stack.pop().unwrap();
        if let Some(parent) = self.stack.last_mut() {
            parent.push_child(Arc::new(node))
        } else {
            self.root = Some(node);
        }
    }

    fn error(&mut self, message: String) {
        self.errors.push(SyntaxError { message, offset: self.pos })
    }

    fn finish(self) -> SyntaxNode {
        SyntaxNode::new(Arc::new(self.root.unwrap()), self.errors)
    }
}


