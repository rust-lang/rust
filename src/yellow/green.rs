use std::sync::Arc;
use {
    SyntaxKind, TextUnit,
    smol_str::SmolStr,
};

#[derive(Clone, Debug)]
pub(crate) enum GreenNode {
    Leaf {
        kind: SyntaxKind,
        text: SmolStr,
    },
    Branch(Arc<GreenBranch>),
}

impl GreenNode {
    pub(crate) fn new_leaf(kind: SyntaxKind, text: &str) -> GreenNode {
        GreenNode::Leaf { kind, text: SmolStr::new(text) }
    }

    pub(crate) fn new_branch(kind: SyntaxKind, children: Box<[GreenNode]>) -> GreenNode {
        GreenNode::Branch(Arc::new(GreenBranch::new(kind, children)))
    }

    pub fn kind(&self) -> SyntaxKind {
        match self {
            GreenNode::Leaf { kind, .. } => *kind,
            GreenNode::Branch(b) => b.kind(),
        }
    }

    pub fn text_len(&self) -> TextUnit {
        match self {
            GreenNode::Leaf { text, ..} => TextUnit::of_str(text.as_str()),
            GreenNode::Branch(b) => b.text_len(),
        }
    }

    pub fn children(&self) -> &[GreenNode] {
        match self {
            GreenNode::Leaf { .. } => &[],
            GreenNode::Branch(b) => b.children(),
        }
    }

    pub fn text(&self) -> String {
        let mut buff = String::new();
        go(self, &mut buff);
        return buff;
        fn go(node: &GreenNode, buff: &mut String) {
            match node {
                GreenNode::Leaf { text, .. } => buff.push_str(text.as_str()),
                GreenNode::Branch(b) => b.children().iter().for_each(|child| go(child, buff)),
            }
        }
    }
}

#[test]
fn assert_send_sync() {
    fn f<T: Send + Sync>() {}
    f::<GreenNode>();
}

#[derive(Clone, Debug)]
pub(crate) struct GreenBranch {
    text_len: TextUnit,
    kind: SyntaxKind,
    children: Box<[GreenNode]>,
}

impl GreenBranch {
    fn new(kind: SyntaxKind, children: Box<[GreenNode]>) -> GreenBranch {
        let text_len = children.iter().map(|x| x.text_len()).sum::<TextUnit>();
        GreenBranch {
            text_len,
            kind,
            children,
        }
    }

    pub fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub fn text_len(&self) -> TextUnit {
        self.text_len
    }

    pub fn children(&self) -> &[GreenNode] {
        &*self.children
    }
}

#[test]
fn test_sizes() {
    use std::mem::size_of;
    println!("GreenBranch = {}", size_of::<GreenBranch>());
    println!("GreenNode   = {}", size_of::<GreenNode>());
    println!("SmolStr     = {}", size_of::<SmolStr>());
}
