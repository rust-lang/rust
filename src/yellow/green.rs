use std::sync::Arc;
use text_unit::TextUnit;
use SyntaxKind;

type TokenText = String;

#[derive(Debug)]
pub(crate) struct GreenNode {
    kind: SyntaxKind,
    data: GreenNodeData,
}

impl GreenNode {
    pub(crate) fn new_leaf(kind: SyntaxKind, text: TokenText) -> GreenNode {
        GreenNode {
            kind,
            data: GreenNodeData::Leaf(GreenLeaf { text }),
        }
    }

    pub(crate) fn new_branch(
        kind: SyntaxKind,
    ) -> GreenNode {
        let branch = GreenBranch {
            text_len: 0.into(),
            leading_trivia: Trivias::default(),
            children: Vec::new(),
        };
        GreenNode {
            kind,
            data: GreenNodeData::Branch(branch),
        }
    }

    pub(crate) fn push_trivia(&mut self, kind: SyntaxKind, text: TokenText) {
        let branch = match &mut self.data {
            GreenNodeData::Branch(branch) => branch,
            _ => panic!()
        };
        branch.text_len += TextUnit::of_str(&text);
        let leading = &mut branch.leading_trivia;
        branch.children.last_mut().map(|(_, t)| t).unwrap_or(leading)
            .push(Arc::new(GreenTrivia { kind, text }));
    }

    pub(crate) fn push_child(&mut self, node: Arc<GreenNode>) {
        let branch = match &mut self.data {
            GreenNodeData::Branch(branch) => branch,
            _ => panic!()
        };
        branch.text_len += node.text_len();
        branch.children.push((node, Trivias::default()));
    }

    pub(crate) fn kind(&self) -> SyntaxKind {
        self.kind
    }

    pub(crate) fn text_len(&self) -> TextUnit {
        match &self.data {
            GreenNodeData::Leaf(l) => l.text_len(),
            GreenNodeData::Branch(b) => b.text_len(),
        }
    }

    pub(crate) fn text(&self) -> String {
        let mut buff = String::new();
        go(self, &mut buff);
        return buff;
        fn go(node: &GreenNode, buff: &mut String) {
            match &node.data {
                GreenNodeData::Leaf(l) => buff.push_str(&l.text),
                GreenNodeData::Branch(branch) => {
                    add_trivia(&branch.leading_trivia, buff);
                    branch.children.iter().for_each(|(child, trivias)| {
                        go(child, buff);
                        add_trivia(trivias, buff);
                    })
                }
            }
        }

        fn add_trivia(trivias: &Trivias, buff: &mut String) {
            trivias.iter().for_each(|t| buff.push_str(&t.text))
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        match &self.data {
            GreenNodeData::Leaf(_) => 0,
            GreenNodeData::Branch(branch) => branch.children.len(),
        }
    }

    pub(crate) fn nth_child(&self, idx: usize) -> &Arc<GreenNode> {
        match &self.data {
            GreenNodeData::Leaf(_) => panic!("leaf nodes have no children"),
            GreenNodeData::Branch(branch) => &branch.children[idx].0,
        }
    }

    pub(crate) fn nth_trivias(&self, idx: usize) -> &Trivias {
        match &self.data {
            GreenNodeData::Leaf(_) => panic!("leaf nodes have no children"),
            GreenNodeData::Branch(branch) => if idx == 0 {
                &branch.leading_trivia
            } else {
                &branch.children[idx - 1].1
            },
        }
    }

    pub(crate) fn is_leaf(&self) -> bool {
        match self.data {
            GreenNodeData::Leaf(_) => true,
            GreenNodeData::Branch(_) => false
        }
    }

    pub(crate) fn leaf_text(&self) -> &str {
        match &self.data {
            GreenNodeData::Leaf(l) => l.text.as_str(),
            GreenNodeData::Branch(_) => panic!("not a leaf")
        }
    }
}

#[derive(Debug)]
enum GreenNodeData {
    Leaf(GreenLeaf),
    Branch(GreenBranch),
}

#[derive(Debug)]
struct GreenLeaf {
    text: TokenText
}

#[derive(Debug)]
struct GreenBranch {
    text_len: TextUnit,
    leading_trivia: Trivias,
    children: Vec<(Arc<GreenNode>, Trivias)>,
}

#[derive(Debug)]
pub(crate) struct GreenTrivia {
    pub(crate) kind: SyntaxKind,
    pub(crate) text: TokenText,
}

type Trivias = Vec<Arc<GreenTrivia>>;


pub(crate) trait TextLen {
    fn text_len(&self) -> TextUnit;
}

impl TextLen for GreenTrivia {
    fn text_len(&self) -> TextUnit {
        TextUnit::of_str(&self.text)
    }
}

impl<T: TextLen> TextLen for Arc<T> {
    fn text_len(&self) -> TextUnit {
        let this: &T = self;
        this.text_len()
    }
}

impl TextLen for GreenNode {
    fn text_len(&self) -> TextUnit {
        self.text_len()
    }
}

impl TextLen for GreenLeaf {
    fn text_len(&self) -> TextUnit {
        TextUnit::of_str(&self.text)
    }
}

impl TextLen for GreenBranch {
    fn text_len(&self) -> TextUnit {
        self.text_len
    }
}

impl<T: TextLen> TextLen for [T] {
    fn text_len(&self) -> TextUnit {
        self.iter().map(TextLen::text_len).sum()
    }
}
