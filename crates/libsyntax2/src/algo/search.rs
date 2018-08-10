use {Node, NodeType, TextUnit, TextRange};
use ::visitor::{visitor, process_subtree_bottom_up};

pub fn child_of_type(node: Node, ty: NodeType) -> Option<Node> {
    node.children().find(|n| n.ty() == ty)
}

pub fn children_of_type<'f>(node: Node<'f>, ty: NodeType) -> Box<Iterator<Item=Node<'f>> + 'f> {
    Box::new(node.children().filter(move |n| n.ty() == ty))
}

pub fn subtree<'f>(node: Node<'f>) -> Box<Iterator<Item=Node<'f>> + 'f> {
    Box::new(node.children().flat_map(subtree).chain(::std::iter::once(node)))
}

pub fn descendants_of_type<'f>(node: Node<'f>, ty: NodeType) -> Vec<Node<'f>> {
    process_subtree_bottom_up(
        node,
        visitor(Vec::new())
            .visit_nodes(&[ty], |node, nodes| nodes.push(node))
    )
}

pub fn child_of_type_exn(node: Node, ty: NodeType) -> Node {
    child_of_type(node, ty).unwrap_or_else(|| {
        panic!("No child of type {:?} for {:?}\
                ----\
                {}\
                ----", ty, node.ty(), node.text())
    })
}


pub fn ancestors(node: Node) -> Ancestors {
    Ancestors(Some(node))
}

pub struct Ancestors<'f>(Option<Node<'f>>);

impl<'f> Iterator for Ancestors<'f> {
    type Item = Node<'f>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.0;
        self.0 = current.and_then(|n| n.parent());
        current
    }
}

pub fn is_leaf(node: Node) -> bool {
    node.children().next().is_none() && !node.range().is_empty()
}


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Direction {
    Left, Right
}

pub fn sibling(node: Node, dir: Direction) -> Option<Node> {
    let (parent, idx) = child_position(node)?;
    let idx = match dir {
        Direction::Left => idx.checked_sub(1)?,
        Direction::Right => idx + 1,
    };
    parent.children().nth(idx)
}

pub mod ast {
    use {Node, AstNode, TextUnit, AstChildren};
    use visitor::{visitor, process_subtree_bottom_up};
    use super::{ancestors, find_leaf_at_offset, LeafAtOffset};

    pub fn ancestor<'f, T: AstNode<'f>>(node: Node<'f>) -> Option<T> {
        ancestors(node)
            .filter_map(T::wrap)
            .next()
    }

    pub fn ancestor_exn<'f, T: AstNode<'f>>(node: Node<'f>) -> T {
        ancestor(node).unwrap()
    }

    pub fn children_of_type<'f, N: AstNode<'f>>(node: Node<'f>) -> AstChildren<N> {
        AstChildren::new(node.children())
    }

    pub fn descendants_of_type<'f, N: AstNode<'f>>(node: Node<'f>) -> Vec<N> {
        process_subtree_bottom_up(
            node,
            visitor(Vec::new())
                .visit::<N, _>(|node, acc| acc.push(node))
        )
    }

    pub fn node_at_offset<'f, T: AstNode<'f>>(node: Node<'f>, offset: TextUnit) -> Option<T> {
        match find_leaf_at_offset(node, offset) {
            LeafAtOffset::None => None,
            LeafAtOffset::Single(node) => ancestor(node),
            LeafAtOffset::Between(left, right) => ancestor(left).or_else(|| ancestor(right)),
        }
    }
}

pub mod traversal {
    use {Node};

    pub fn bottom_up<'f, F: FnMut(Node<'f>)>(node: Node<'f>, mut f: F)
    {
        go(node, &mut f);

        fn go<'f, F: FnMut(Node<'f>)>(node: Node<'f>, f: &mut F) {
            for child in node.children() {
                go(child, f)
            }
            f(node);
        }
    }
}

fn child_position(child: Node) -> Option<(Node, usize)> {
    child.parent()
        .map(|parent| {
            (parent, parent.children().position(|n| n == child).unwrap())
        })
}

fn common_ancestor<'f>(n1: Node<'f>, n2: Node<'f>) -> Node<'f> {
    for p in ancestors(n1) {
        if ancestors(n2).any(|a| a == p) {
            return p;
        }
    }
    panic!("Can't find common ancestor of {:?} and {:?}", n1, n2)
}

