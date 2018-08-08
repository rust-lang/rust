use std::sync::Arc;
use {
    SyntaxKind, TextUnit,
    smol_str::SmolStr,
};

#[derive(Clone, Debug)]
pub(crate) enum GreenNode {
    Leaf(GreenLeaf),
    Branch(Arc<GreenBranch>),
}

impl GreenNode {
    pub(crate) fn new_leaf(kind: SyntaxKind, text: &str) -> GreenNode {
        GreenNode::Leaf(GreenLeaf::new(kind, text))
    }

    pub(crate) fn new_branch(kind: SyntaxKind, children: Box<[GreenNode]>) -> GreenNode {
        GreenNode::Branch(Arc::new(GreenBranch::new(kind, children)))
    }

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

#[derive(Clone, Debug)]
pub(crate) struct GreenLeaf {
    kind: SyntaxKind,
    text: SmolStr,
}

impl GreenLeaf {
    fn new(kind: SyntaxKind, text: &str) -> Self {
        let text = SmolStr::new(text);
        GreenLeaf { kind, text }
    }

    pub(crate) fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub(crate) fn text(&self) -> &str {
        self.text.as_str()
    }

    pub(crate) fn text_len(&self) -> TextUnit {
        TextUnit::of_str(self.text())
    }
}


#[test]
fn test_sizes() {
    use std::mem::size_of;

    println!("GreenNode = {}", size_of::<GreenNode>());
    println!("GreenLeaf = {}", size_of::<GreenLeaf>());
    println!("SyntaxKind = {}", size_of::<SyntaxKind>());
    println!("SmolStr = {}", size_of::<SmolStr>());
}
