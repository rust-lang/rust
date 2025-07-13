pub struct BTree<V> {
    pub node: TreeItem<V>,
}

pub enum TreeItem<V> {
    TreeLeaf { value: V },
}

pub fn leaf<V>(value: V) -> TreeItem<V> {
    TreeItem::TreeLeaf { value: value }
}

fn main() {
    BTree::<isize> { node: leaf(1) };
}
