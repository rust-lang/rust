// compile-flags: -Z parse-only

enum BtNode {
    Node(u32,Box<BtNode>,Box<BtNode>),
    Leaf(u32),
}

fn main() {
    let y = match x {
        Foo<T>::A(value) => value, //~ error: expected one of `=>`, `@`, `if`, or `|`, found `<`
        Foo<T>::B => 7,
    };
}
