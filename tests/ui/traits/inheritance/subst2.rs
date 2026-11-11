//@ run-pass

trait Panda<T> {
    fn chomp(&self, bamboo: &T) -> T;
}

trait Add<RHS,Result>: Panda<RHS> {
    fn add(&self, rhs: &RHS) -> Result;
}

trait MyNum : Sized + Add<Self,Self> { }

struct MyInt { val: isize }

impl Panda<MyInt> for MyInt {
    fn chomp(&self, bamboo: &MyInt) -> MyInt {
        mi(self.val + bamboo.val)
    }
}

impl Add<MyInt, MyInt> for MyInt {
    fn add(&self, other: &MyInt) -> MyInt { self.chomp(other) }
}

impl MyNum for MyInt {}

fn f<T:MyNum>(x: T, y: T) -> T {
    return x.add(&y).chomp(&y);
}

fn mi(v: isize) -> MyInt { MyInt { val: v } }

pub fn main() {
    let (x, y) = (mi(3), mi(5));
    let z = f(x, y);
    assert_eq!(z.val, 13);
}
