use {SyntaxKind, TextUnit, TextRange};
use super::{NodeData, NodeIdx, File};

pub struct FileBuilder {
    text: String,
    nodes: Vec<NodeData>,
    in_progress: Vec<(NodeIdx, Option<NodeIdx>)>, // (parent, last_child)
    pos: TextUnit,
}

impl FileBuilder {
    pub fn new(text: String) -> FileBuilder {
        FileBuilder {
            text,
            nodes: Vec::new(),
            in_progress: Vec::new(),
            pos: TextUnit::new(0),
        }
    }

    pub fn finish(self) -> File {
        assert!(self.in_progress.is_empty());
        assert!(self.pos == (self.text.len() as u32).into());
        File {
            text: self.text,
            nodes: self.nodes,
        }
    }

    pub fn leaf(&mut self, kind: SyntaxKind, len: TextUnit) {
        let leaf = NodeData {
            kind,
            range: TextRange::from_len(self.pos, len),
            parent: None,
            first_child: None,
            next_sibling: None,
        };
        self.pos += len;
        let id = self.push_child(leaf);
        self.add_len(id);
    }

    pub fn start_internal(&mut self, kind: SyntaxKind) {
        let node = NodeData {
            kind,
            range: TextRange::from_len(self.pos, 0.into()),
            parent: None,
            first_child: None,
            next_sibling: None,
        };
        let id = if self.in_progress.is_empty() {
            self.new_node(node)
        } else {
            self.push_child(node)
        };
        self.in_progress.push((id, None))
    }

    pub fn finish_internal(&mut self) {
        let (id, _) = self.in_progress.pop().unwrap();
        if !self.in_progress.is_empty() {
            self.add_len(id);
        }
    }

    fn new_node(&mut self, data: NodeData) -> NodeIdx {
        let id = NodeIdx(self.nodes.len() as u32);
        self.nodes.push(data);
        id
    }

    fn push_child(&mut self, mut child: NodeData) -> NodeIdx {
        child.parent = Some(self.current_id());
        let id = self.new_node(child);
        if let Some(sibling) = self.current_sibling() {
            fill(&mut sibling.next_sibling, id);
            return id
        }
        fill(&mut self.current_parent().first_child, id);
        id
    }

    fn add_len(&mut self, child: NodeIdx) {
        let range = self.nodes[child.0 as usize].range;
        grow(&mut self.current_parent().range, range);
    }

    fn current_id(&self) -> NodeIdx {
        self.in_progress.last().unwrap().0
    }

    fn current_parent(&mut self) -> &mut NodeData {
        let NodeIdx(idx) = self.current_id();
        &mut self.nodes[idx as usize]
    }

    fn current_sibling(&mut self) -> Option<&mut NodeData> {
        let NodeIdx(idx) = self.in_progress.last().unwrap().1?;
        Some(&mut self.nodes[idx as usize])
    }
}

fn fill<T>(slot: &mut Option<T>, value: T) {
    assert!(slot.is_none());
    *slot = Some(value);
}

fn grow(left: &mut TextRange, right: TextRange) {
    assert_eq!(left.end(), right.start());
    *left = TextRange::from_to(left.start(), right.end())
}