enum BtNode {
    Node(u32,Box<BtNode>,Box<BtNode>),
    Leaf(u32),
}

fn main() {
    let y = match 10 {
        Foo<T>::A(value) => value, //~ ERROR generic args in patterns require the turbofish syntax
        Foo<T>::B => 7,
    };
}
