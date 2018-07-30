use {
    parser::Sink,
    yellow::{GreenNode, GreenNodeBuilder, SyntaxError, SyntaxNode, SyntaxRoot},
    SyntaxKind, TextRange, TextUnit,
};

pub(crate) struct GreenBuilder {
    text: String,
    stack: Vec<GreenNodeBuilder>,
    pos: TextUnit,
    root: Option<GreenNode>,
    errors: Vec<SyntaxError>,
}

impl GreenBuilder {}

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
        let text = &self.text[range];
        let leaf = GreenNodeBuilder::new_leaf(kind, text);
        let parent = self.stack.last_mut().unwrap();
        parent.push_child(leaf)
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        self.stack.push(GreenNodeBuilder::new_internal(kind))
    }

    fn finish_internal(&mut self) {
        let builder = self.stack.pop().unwrap();
        let node = builder.build();
        if let Some(parent) = self.stack.last_mut() {
            parent.push_child(node);
        } else {
            self.root = Some(node);
        }
    }

    fn error(&mut self, message: String) {
        self.errors.push(SyntaxError {
            message,
            offset: self.pos,
        })
    }

    fn finish(self) -> SyntaxNode {
        let root = SyntaxRoot::new(self.root.unwrap(), self.errors);
        SyntaxNode::new_owned(root)
    }
}
