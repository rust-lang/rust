//@ run-pass


trait A {
    fn g(&self) -> isize { 10 }
}

impl A for isize { }

fn f<T:A>(i: T) {
    assert_eq!(i.g(), 10);
}

pub fn main () {
    f(0);
}
