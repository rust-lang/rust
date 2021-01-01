enum BtNode {
    Node(u32,Box<BtNode>,Box<BtNode>),
    Leaf(u32),
}

fn main() {
    let y = match 10 {
        Foo<T>::A(value) => value, //~ ERROR failed to resolve: use of undeclared type `Foo`
        //~^ ERROR cannot find type `T` in this scope
        Foo<T>::B => 7, //~ ERROR failed to resolve: use of undeclared type `Foo`
        //~^ ERROR cannot find type `T` in this scope
    };
}
