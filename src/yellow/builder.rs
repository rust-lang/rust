use {
    parser_impl::Sink,
    yellow::{GreenNode, SyntaxError, SyntaxNode, SyntaxRoot},
    SyntaxKind, TextRange, TextUnit,
};

pub(crate) struct GreenBuilder<'a> {
    text: &'a str,
    parents: Vec<(SyntaxKind, usize)>,
    children: Vec<GreenNode>,
    pos: TextUnit,
    errors: Vec<SyntaxError>,
}

impl<'a> Sink<'a> for GreenBuilder<'a> {
    type Tree = SyntaxNode;

    fn new(text: &'a str) -> Self {
        GreenBuilder {
            text,
            parents: Vec::new(),
            children: Vec::new(),
            pos: 0.into(),
            errors: Vec::new(),
        }
    }

    fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let range = TextRange::offset_len(self.pos, len);
        self.pos += len;
        let text = &self.text[range];
        self.children.push(
            GreenNode::new_leaf(kind, text)
        );
    }

    fn start_internal(&mut self, kind: SyntaxKind) {
        let len = self.children.len();
        self.parents.push((kind, len));
    }

    fn finish_internal(&mut self) {
        let (kind, first_child) = self.parents.pop().unwrap();
        let children: Vec<_> = self.children
            .drain(first_child..)
            .collect();
        self.children.push(
            GreenNode::new_branch(kind, children.into_boxed_slice())
        );
    }

    fn error(&mut self, message: String) {
        self.errors.push(SyntaxError {
            message,
            offset: self.pos,
        })
    }

    fn finish(mut self) -> SyntaxNode {
        assert_eq!(self.children.len(), 1);
        let root = self.children.pop().unwrap();
        let root = SyntaxRoot::new(root, self.errors);
        SyntaxNode::new_owned(root)
    }
}
