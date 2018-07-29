use std::{
    ptr,
    sync::RwLock,
};
use {
    TextUnit,
    yellow::GreenNode,
};

#[derive(Debug)]
pub(crate) struct RedNode {
    green: GreenNode,
    parent: Option<ParentData>,
    children: RwLock<Vec<Option<Box<RedNode>>>>,
}

#[derive(Debug)]
struct ParentData {
    parent: *const RedNode,
    start_offset: TextUnit,
    index_in_parent: usize,
}

impl RedNode {
    pub fn new_root(
        green: GreenNode,
    ) -> RedNode {
        RedNode::new(green, None)
    }

    fn new_child(
        green: GreenNode,
        parent: *const RedNode,
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

    fn new(
        green: GreenNode,
        parent: Option<ParentData>,
    ) -> RedNode {
        let n_children = green.children().len();
        let children = (0..n_children).map(|_| None).collect();
        RedNode { green, parent, children: RwLock::new(children) }
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

    pub(crate) fn nth_child(&self, idx: usize) -> ptr::NonNull<RedNode> {
        match &self.children.read().unwrap()[idx] {
            Some(child) => return ptr::NonNull::from(&**child),
            None => (),
        }
        let mut children = self.children.write().unwrap();
        if children[idx].is_none() {
            let green_children = self.green.children();
            let start_offset = self.start_offset()
                + green_children[..idx].iter().map(|x| x.text_len()).sum::<TextUnit>();
            let child = RedNode::new_child(green_children[idx].clone(), self, start_offset, idx);
            children[idx] = Some(Box::new(child))
        }
        let child = children[idx].as_ref().unwrap();
        ptr::NonNull::from(&**child)
    }
}
