use std::sync::{Arc, RwLock};
use {
    TextUnit,
    yellow::{Ptr, GreenNode},
};

#[derive(Debug)]
pub(crate) struct RedNode {
    green: GreenNode,
    parent: Option<ParentData>,
    children: RwLock<Vec<Option<Arc<RedNode>>>>,
}

#[derive(Debug)]
struct ParentData {
    parent: Ptr<RedNode>,
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
        parent: Ptr<RedNode>,
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
        let children = vec![None; green.children().len()];
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

    pub(crate) fn nth_child(&self, me: Ptr<RedNode>, idx: usize) -> Arc<RedNode> {
        match &self.children.read().unwrap()[idx] {
            Some(child) => return child.clone(),
            None => (),
        }
        let mut children = self.children.write().unwrap();
        if children[idx].is_none() {
            let green_children = self.green.children();
            let start_offset = self.start_offset()
                + green_children[..idx].iter().map(|x| x.text_len()).sum::<TextUnit>();
            let child = RedNode::new_child(green_children[idx].clone(), me, start_offset, idx);
            children[idx] = Some(Arc::new(child))
        }
        children[idx].as_ref().unwrap().clone()
    }
}
