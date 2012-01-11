// Test cyclic detector when using iface instances.

tag Tree = TreeR;
type TreeR = @{
    mutable left: option<Tree>,
    mutable right: option<Tree>,
    val: to_str
};

iface to_str {
    fn to_str() -> str;
}

impl <T: to_str> of to_str for option<T> {
    fn to_str() -> str {
        alt self {
          none. { "none" }
          some(t) { "some(" + t.to_str() + ")" }
        }
    }
}

impl of to_str for int {
    fn to_str() -> str { int::str(self) }
}

impl of to_str for Tree {
    fn to_str() -> str {
        #fmt["[%s, %s, %s]",
             self.val.to_str(),
             self.left.to_str(),
             self.right.to_str()]
    }
}

fn main() {
    let t1 = Tree(@{mutable left: none,
                    mutable right: none,
                    val: 1 as to_str });
    let t2 = Tree(@{mutable left: some(t1),
                    mutable right: some(t1),
                    val: 2 as to_str });
    assert t2.to_str() == "[2, some([1, none, none]), some([1, none, none])]";
    t1.left = some(t2); // create cycle
}
