use parking_lot::RwLock;
use {yellow::{GreenNode, RedPtr}, TextUnit};

#[derive(Debug)]
pub(crate) struct RedNode {
    green: GreenNode,
    parent: Option<ParentData>,
    children: RwLock<Box<[RedChild]>>,
}

#[derive(Debug)]
enum RedChild {
    Zigot(TextUnit),
    Child(RedNode)
}

impl RedChild {
    fn set(&mut self, node: RedNode) -> &RedNode {
        match self {
            RedChild::Child(node) => return node,
            RedChild::Zigot(_) => {
                *self = RedChild::Child(node);
                match self {
                    RedChild::Child(node) => return node,
                    RedChild::Zigot(_) => unreachable!()
                }
            }
        }
    }
}

#[derive(Debug)]
struct ParentData {
    parent: RedPtr,
    start_offset: TextUnit,
    index_in_parent: usize,
}

impl RedNode {
    pub fn new_root(green: GreenNode) -> RedNode {
        RedNode::new(green, None)
    }

    fn new_child(
        green: GreenNode,
        parent: RedPtr,
        start_offset: TextUnit,
        index_in_parent: usize,
    ) -> RedNode {
        let parent_data = ParentData {
            parent,
            start_offset,
            index_in_parent,
        };
        RedNode::new(green, Some(parent_data))
    }

    fn new(green: GreenNode, parent: Option<ParentData>) -> RedNode {
        let start_offset = parent.as_ref().map(|it| it.start_offset).unwrap_or(0.into());
        let children = green.children()
            .iter()
            .scan(start_offset, |start_offset, child| {
                let res = RedChild::Zigot(*start_offset);
                *start_offset += child.text_len();
                Some(res)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        RedNode {
            green,
            parent,
            children: RwLock::new(children),
        }
    }

    pub(crate) fn green(&self) -> &GreenNode {
        &self.green
    }

    pub(crate) fn start_offset(&self) -> TextUnit {
        match &self.parent {
            None => 0.into(),
            Some(p) => p.start_offset,
        }
    }

    pub(crate) fn n_children(&self) -> usize {
        self.green.children().len()
    }

    pub(crate) fn get_child(&self, idx: usize) -> Option<RedPtr> {
        if idx >= self.n_children() {
            return None;
        }
        let start_offset = match &self.children.read()[idx] {
            RedChild::Child(child) => return Some(RedPtr::new(child)),
            RedChild::Zigot(start_offset) => *start_offset,
        };
        let green_children = self.green.children();
        let child =
            RedNode::new_child(green_children[idx].clone(), RedPtr::new(self), start_offset, idx);
        let mut children = self.children.write();
        let child = children[idx].set(child);
        Some(RedPtr::new(child))
    }

    pub(crate) fn parent(&self) -> Option<RedPtr> {
        Some(self.parent.as_ref()?.parent)
    }
    pub(crate) fn index_in_parent(&self) -> Option<usize> {
        Some(self.parent.as_ref()?.index_in_parent)
    }
}
