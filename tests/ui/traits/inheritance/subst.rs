// run-pass

pub trait Add<RHS,Result> {
    fn add(&self, rhs: &RHS) -> Result;
}

trait MyNum : Sized + Add<Self,Self> { }

struct MyInt { val: isize }

impl Add<MyInt, MyInt> for MyInt {
    fn add(&self, other: &MyInt) -> MyInt { mi(self.val + other.val) }
}

impl MyNum for MyInt {}

fn f<T:MyNum>(x: T, y: T) -> T {
    return x.add(&y);
}

fn mi(v: isize) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y) = (mi(3), mi(5));
    let z = f(x, y);
    assert_eq!(z.val, 8)
}
