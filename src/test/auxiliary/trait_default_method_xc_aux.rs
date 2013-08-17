
pub struct Something { x: int }

pub trait A {
    fn f(&self) -> int;
    fn g(&self) -> int { 10 }
    fn h(&self) -> int { 11 }
    fn lurr(x: &Self, y: &Self) -> int { x.g() + y.h() }
}


impl A for int {
    fn f(&self) -> int { 10 }
}

impl A for Something {
    fn f(&self) -> int { 10 }
}

trait B<T> {
    fn thing<U>(&self, x: T, y: U) -> (T, U) { (x, y) }
    fn staticthing<U>(_z: &Self, x: T, y: U) -> (T, U) { (x, y) }
}

impl<T> B<T> for int { }
impl B<float> for bool { }



pub trait TestEquality {
    fn test_eq(&self, rhs: &Self) -> bool;
    fn test_neq(&self, rhs: &Self) -> bool {
        !self.test_eq(rhs)
    }
}

impl TestEquality for int {
    fn test_eq(&self, rhs: &int) -> bool {
        *self == *rhs
    }
}
