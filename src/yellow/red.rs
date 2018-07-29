use std::sync::{Arc, RwLock};
use {
    TextUnit,
    yellow::{Ptr, GreenNode, TextLen}
};

#[derive(Debug)]
pub(crate) struct RedNode {
    green: Arc<GreenNode>,
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
        green: Arc<GreenNode>,
    ) -> RedNode {
        RedNode::new(green, None)
    }

    fn new_child(
        green: Arc<GreenNode>,
        parent: Ptr<RedNode>,
        start_offset: TextUnit,
        index_in_parent: usize
    ) -> RedNode {
        let parent_data = ParentData {
            parent,
            start_offset,
            index_in_parent
        };
        RedNode::new(green, Some(parent_data))
    }

    fn new(
        green: Arc<GreenNode>,
        parent: Option<ParentData>,
    ) -> RedNode {
        let children = vec![None; green.n_children()];
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
        self.green.n_children()
    }

    pub(crate) fn nth_child(&self, me: Ptr<RedNode>, n: usize) -> Arc<RedNode> {
        match &self.children.read().unwrap()[n] {
            Some(child) => return child.clone(),
            None => (),
        }
        let mut children = self.children.write().unwrap();
        if children[n].is_none() {
            let start_offset = {
                let mut acc = self.start_offset();
                for i in 0..n {
                    acc += self.green.nth_trivias(i).text_len();
                    acc += self.green.nth_child(i).text_len();
                }
                acc += self.green.nth_trivias(n).text_len();
                acc
            };
            let green = self.green.nth_child(n).clone();
            let child = RedNode::new_child(green, me, start_offset, n);
            children[n] = Some(Arc::new(child))
        }
        children[n].as_ref().unwrap().clone()
    }
}
