use std::{ptr, sync::RwLock};
use {yellow::GreenNode, TextUnit};

#[derive(Debug)]
pub(crate) struct RedNode {
    green: GreenNode,
    parent: Option<ParentData>,
    children: RwLock<Box<[Option<RedNode>]>>,
}

#[derive(Debug)]
struct ParentData {
    parent: ptr::NonNull<RedNode>,
    start_offset: TextUnit,
    index_in_parent: usize,
}

impl RedNode {
    pub fn new_root(green: GreenNode) -> RedNode {
        RedNode::new(green, None)
    }

    fn new_child(
        green: GreenNode,
        parent: ptr::NonNull<RedNode>,
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
        let n_children = green.children().len();
        let children = (0..n_children)
            .map(|_| None)
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

    pub(crate) fn get_child(&self, idx: usize) -> Option<ptr::NonNull<RedNode>> {
        if idx >= self.n_children() {
            return None;
        }
        match &self.children.read().unwrap()[idx] {
            Some(child) => return Some(child.into()),
            None => (),
        }
        let mut children = self.children.write().unwrap();
        if children[idx].is_none() {
            let green_children = self.green.children();
            let start_offset = self.start_offset()
                + green_children[..idx]
                    .iter()
                    .map(|x| x.text_len())
                    .sum::<TextUnit>();
            let child =
                RedNode::new_child(green_children[idx].clone(), self.into(), start_offset, idx);
            children[idx] = Some(child)
        }
        Some(children[idx].as_ref().unwrap().into())
    }

    pub(crate) fn parent(&self) -> Option<ptr::NonNull<RedNode>> {
        Some(self.parent.as_ref()?.parent)
    }
    pub(crate) fn index_in_parent(&self) -> Option<usize> {
        Some(self.parent.as_ref()?.index_in_parent)
    }
}
