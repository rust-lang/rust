// Test cyclic detector when using iface instances.

enum Tree = TreeR;
type TreeR = @{
    mut left: option<Tree>,
    mut right: option<Tree>,
    val: to_str
};

iface to_str {
    fn to_str() -> ~str;
}

impl <T: to_str> of to_str for option<T> {
    fn to_str() -> ~str {
        alt self {
          none { ~"none" }
          some(t) { ~"some(" + t.to_str() + ~")" }
        }
    }
}

impl of to_str for int {
    fn to_str() -> ~str { int::str(self) }
}

impl of to_str for Tree {
    fn to_str() -> ~str {
        let l = self.left, r = self.right;
        fmt!{"[%s, %s, %s]", self.val.to_str(),
             l.to_str(), r.to_str()}
    }
}

fn foo<T: to_str>(x: T) -> ~str { x.to_str() }

fn main() {
    let t1 = Tree(@{mut left: none,
                    mut right: none,
                    val: 1 as to_str });
    let t2 = Tree(@{mut left: some(t1),
                    mut right: some(t1),
                    val: 2 as to_str });
    let expected = ~"[2, some([1, none, none]), some([1, none, none])]";
    assert t2.to_str() == expected;
    assert foo(t2 as to_str) == expected;
    t1.left = some(t2); // create cycle
}
