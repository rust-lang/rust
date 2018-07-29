use std::sync::Arc;
use {SyntaxKind::{self, *}, TextUnit};

#[derive(Clone, Debug)]
pub(crate) enum GreenNode {
    Leaf(GreenLeaf),
    Branch(Arc<GreenBranch>),
}

impl GreenNode {
    pub fn kind(&self) -> SyntaxKind {
        match self {
            GreenNode::Leaf(l) => l.kind(),
            GreenNode::Branch(b) => b.kind(),
        }
    }

    pub fn text_len(&self) -> TextUnit {
        match self {
            GreenNode::Leaf(l) => l.text_len(),
            GreenNode::Branch(b) => b.text_len(),
        }
    }

    pub fn children(&self) -> &[GreenNode] {
        match self {
            GreenNode::Leaf(_) => &[],
            GreenNode::Branch(b) => b.children(),
        }
    }

    pub fn text(&self) -> String {
        let mut buff = String::new();
        go(self, &mut buff);
        return buff;
        fn go(node: &GreenNode, buff: &mut String) {
            match node {
                GreenNode::Leaf(l) => buff.push_str(&l.text()),
                GreenNode::Branch(b) => {
                    b.children().iter().for_each(|child| go(child, buff))
                }
            }
        }
    }
}

pub(crate) struct GreenNodeBuilder {
    kind: SyntaxKind,
    children: Vec<GreenNode>,
}

impl GreenNodeBuilder {
    pub(crate) fn new_leaf(kind: SyntaxKind, text: &str) -> GreenNode {
        GreenNode::Leaf(GreenLeaf::new(kind, text))
    }

    pub(crate) fn new_internal(kind: SyntaxKind) -> GreenNodeBuilder {
        GreenNodeBuilder {
            kind,
            children: Vec::new(),
        }
    }

    pub(crate) fn push_child(&mut self, node: GreenNode) {
        self.children.push(node)
    }

    pub(crate) fn build(self) -> GreenNode {
        let branch = GreenBranch::new(self.kind, self.children);
        GreenNode::Branch(Arc::new(branch))
    }
}


#[test]
fn assert_send_sync() {
    fn f<T: Send + Sync>() {}
    f::<GreenNode>();
}

#[derive(Clone, Debug)]
pub(crate) enum GreenLeaf {
    Whitespace {
        newlines: u8,
        spaces: u8,
    },
    Token {
        kind: SyntaxKind,
        text: Arc<str>,
    },
}

impl GreenLeaf {
    fn new(kind: SyntaxKind, text: &str) -> Self {
        if kind == WHITESPACE {
            let newlines = text.bytes().take_while(|&b| b == b'\n').count();
            let spaces = text[newlines..].bytes().take_while(|&b| b == b' ').count();
            if newlines + spaces == text.len() && newlines <= N_NEWLINES && spaces <= N_SPACES {
                return GreenLeaf::Whitespace { newlines: newlines as u8, spaces: spaces as u8 };
            }
        }
        GreenLeaf::Token { kind, text: text.to_owned().into_boxed_str().into() }
    }

    pub(crate) fn kind(&self) -> SyntaxKind {
        match self {
            GreenLeaf::Whitespace { .. } => WHITESPACE,
            GreenLeaf::Token { kind, .. } => *kind,
        }
    }

    pub(crate) fn text(&self) -> &str {
        match self {
            &GreenLeaf::Whitespace { newlines, spaces } => {
                let newlines = newlines as usize;
                let spaces = spaces as usize;
                assert!(newlines <= N_NEWLINES && spaces <= N_SPACES);
                &WS[N_NEWLINES - newlines..N_NEWLINES + spaces]
            }
            GreenLeaf::Token { text, .. } => text,
        }
    }

    pub(crate) fn text_len(&self) -> TextUnit {
        TextUnit::of_str(self.text())
    }
}

const N_NEWLINES: usize = 16;
const N_SPACES: usize = 64;
const WS: &str =
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n                                                                ";

#[derive(Clone, Debug)]
pub(crate) struct GreenBranch {
    text_len: TextUnit,
    kind: SyntaxKind,
    children: Vec<GreenNode>,
}

impl GreenBranch {
    fn new(kind: SyntaxKind, children: Vec<GreenNode>) -> GreenBranch {
        let text_len = children.iter().map(|x| x.text_len()).sum::<TextUnit>();
        GreenBranch { text_len, kind, children }
    }

    pub fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub fn text_len(&self) -> TextUnit {
        self.text_len
    }

    pub fn children(&self) -> &[GreenNode] {
        self.children.as_slice()
    }
}

