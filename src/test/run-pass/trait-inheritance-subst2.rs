trait Panda<T> {
    fn chomp(bamboo: &T) -> T;
}

trait Add<RHS,Result>: Panda<RHS> {
    fn add(rhs: &RHS) -> Result;
}

trait MyNum : Add<self,self> { }

struct MyInt { val: int }

impl MyInt : Panda<MyInt> {
    fn chomp(bamboo: &MyInt) -> MyInt {
        mi(self.val + bamboo.val)
    }
}

impl MyInt : Add<MyInt, MyInt> {
    fn add(other: &MyInt) -> MyInt { self.chomp(other) }
}

impl MyInt : MyNum;

fn f<T:MyNum>(x: T, y: T) -> T {
    return x.add(&y).chomp(&y);
}

fn mi(v: int) -> MyInt { MyInt { val: v } }

fn main() {
    let (x, y) = (mi(3), mi(5));
    let z = f(x, y);
    assert z.val == 13;
}

