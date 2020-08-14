//! A simple tree implementation which tries to not allocate all over the place.
use std::ops;

use arena::Arena;

#[derive(Default)]
pub struct Tree<T> {
    nodes: Arena<Node<T>>,
    current_path: Vec<(Idx<T>, Option<Idx<T>>)>,
}

pub type Idx<T> = arena::Idx<Node<T>>;

impl<T> Tree<T> {
    pub fn start(&mut self)
    where
        T: Default,
    {
        let me = self.nodes.alloc(Node::new(T::default()));
        if let Some((parent, last_child)) = self.current_path.last_mut() {
            let slot = match *last_child {
                Some(last_child) => &mut self.nodes[last_child].next_sibling,
                None => &mut self.nodes[*parent].first_child,
            };
            let prev = slot.replace(me);
            assert!(prev.is_none());
            *last_child = Some(me);
        }

        self.current_path.push((me, None));
    }

    pub fn finish(&mut self, data: T) {
        let (me, _last_child) = self.current_path.pop().unwrap();
        self.nodes[me].data = data;
    }

    pub fn root(&self) -> Option<Idx<T>> {
        self.nodes.iter().next().map(|(idx, _)| idx)
    }

    pub fn children(&self, idx: Idx<T>) -> impl Iterator<Item = Idx<T>> + '_ {
        NodeIter { nodes: &self.nodes, next: self.nodes[idx].first_child }
    }
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.current_path.clear();
    }
}

impl<T> ops::Index<Idx<T>> for Tree<T> {
    type Output = T;
    fn index(&self, index: Idx<T>) -> &T {
        &self.nodes[index].data
    }
}

pub struct Node<T> {
    data: T,
    first_child: Option<Idx<T>>,
    next_sibling: Option<Idx<T>>,
}

impl<T> Node<T> {
    fn new(data: T) -> Node<T> {
        Node { data, first_child: None, next_sibling: None }
    }
}

struct NodeIter<'a, T> {
    nodes: &'a Arena<Node<T>>,
    next: Option<Idx<T>>,
}

impl<'a, T> Iterator for NodeIter<'a, T> {
    type Item = Idx<T>;

    fn next(&mut self) -> Option<Idx<T>> {
        self.next.map(|next| {
            self.next = self.nodes[next].next_sibling;
            next
        })
    }
}
