//@ run-pass


trait A {
    fn g<T>(&self, x: T, y: T) -> (T, T) { (x, y) }
}

impl A for isize { }

fn f<T, V: A>(i: V, j: T, k: T) -> (T, T) {
    i.g(j, k)
}

pub fn main () {
    assert_eq!(f(0, 1, 2), (1, 2));
    assert_eq!(f(0, 1u8, 2u8), (1u8, 2u8));
}
