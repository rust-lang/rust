// run-pass
trait Foo<T> {
    fn f(&self, x: &T);
}

trait Bar : Sized + Foo<Self> {
    fn g(&self);
}

struct S {
    x: isize
}

impl Foo<S> for S {
    fn f(&self, x: &S) {
        println!("{}", x.x);
    }
}

impl Bar for S {
    fn g(&self) {
        self.f(self);
    }
}

pub fn main() {
    let s = S { x: 1 };
    s.g();
}
