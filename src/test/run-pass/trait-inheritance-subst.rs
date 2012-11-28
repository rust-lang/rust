pub trait Add<RHS,Result> {
    pure fn add(rhs: &RHS) -> Result;
}

trait MyNum : Add<self,self> { }

struct MyInt { val: int }

impl MyInt : Add<MyInt, MyInt> {
    pure fn add(other: &MyInt) -> MyInt { mi(self.val + other.val) }
}

impl MyInt : MyNum;

fn f<T:MyNum>(x: T, y: T) -> T {
    return x.add(&y);
}

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

fn main() {
    let (x, y) = (mi(3), mi(5));
    let z = f(x, y);
    assert z.val == 8
}

