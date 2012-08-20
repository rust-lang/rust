// Test cyclic detector when using trait instances.

enum Tree = TreeR;
type TreeR = @{
    mut left: Option<Tree>,
    mut right: Option<Tree>,
    val: to_str
};

trait to_str {
    fn to_str() -> ~str;
}

impl <T: to_str> Option<T>: to_str {
    fn to_str() -> ~str {
        match self {
          None => { ~"none" }
          Some(t) => { ~"some(" + t.to_str() + ~")" }
        }
    }
}

impl int: to_str {
    fn to_str() -> ~str { int::str(self) }
}

impl Tree: to_str {
    fn to_str() -> ~str {
        let l = self.left, r = self.right;
        fmt!("[%s, %s, %s]", self.val.to_str(),
             l.to_str(), r.to_str())
    }
}

fn foo<T: to_str>(x: T) -> ~str { x.to_str() }

fn main() {
    let t1 = Tree(@{mut left: None,
                    mut right: None,
                    val: 1 as to_str });
    let t2 = Tree(@{mut left: Some(t1),
                    mut right: Some(t1),
                    val: 2 as to_str });
    let expected = ~"[2, some([1, none, none]), some([1, none, none])]";
    assert t2.to_str() == expected;
    assert foo(t2 as to_str) == expected;
    t1.left = Some(t2); // create cycle
}
