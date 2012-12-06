use cmp::Eq;

pub trait MyNum : Add<self,self>, Sub<self,self>, Mul<self,self>, Eq {
}

pub struct MyInt {
    val: int
}

pub impl MyInt : Add<MyInt, MyInt> {
    pure fn add(&self, other: &MyInt) -> MyInt { mi(self.val + other.val) }
}

pub impl MyInt : Sub<MyInt, MyInt> {
    pure fn sub(&self, other: &MyInt) -> MyInt { mi(self.val - other.val) }
}

pub impl MyInt : Mul<MyInt, MyInt> {
    pure fn mul(&self, other: &MyInt) -> MyInt { mi(self.val * other.val) }
}

pub impl MyInt : Eq {
    pure fn eq(&self, other: &MyInt) -> bool { self.val == other.val }

    pure fn ne(&self, other: &MyInt) -> bool { !self.eq(other) }
}

pub impl MyInt : MyNum;

pure fn mi(v: int) -> MyInt { MyInt { val: v } }

