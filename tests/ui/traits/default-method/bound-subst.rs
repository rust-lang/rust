//@ run-pass


trait A<T> {
    fn g<U>(&self, x: T, y: U) -> (T, U) { (x, y) }
}

impl A<i32> for i32 { }
impl<T> A<T> for u32 { }

fn f<T, U, V: A<T>>(i: V, j: T, k: U) -> (T, U) {
    i.g(j, k)
}

pub fn main () {
    assert_eq!(f(0, 1, 2), (1, 2));
    assert_eq!(f(0, 1, 2), (1, 2));
}
