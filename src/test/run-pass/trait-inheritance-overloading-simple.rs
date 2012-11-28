use cmp::Eq;

trait MyNum : Eq { }

struct MyInt { val: int }

impl MyInt : Eq {
    pure fn eq(&self, other: &MyInt) -> bool { self.val == other.val }
    pure fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

impl MyInt : MyNum;

fn f<T:MyNum>(x: T, y: T) -> bool {
    return x == y;
}

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

fn main() {
    let (x, y, z) = (mi(3), mi(5), mi(3));
    assert x != y;
    assert x == z;
}

