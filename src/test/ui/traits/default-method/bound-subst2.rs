// run-pass


trait A<T> {
    fn g(&self, x: T) -> T { x }
}

impl A<isize> for isize { }

fn f<T, V: A<T>>(i: V, j: T) -> T {
    i.g(j)
}

pub fn main () {
    assert_eq!(f(0, 2), 2);
}
