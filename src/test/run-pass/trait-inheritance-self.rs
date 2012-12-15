trait Foo<T> {
    fn f(&self, x: &T);
}

trait Bar : Foo<self> {
    fn g(&self);
}

struct S {
    x: int
}

impl S : Foo<S> {
    fn f(&self, x: &S) {
        io::println(x.x.to_str());
    }
}

impl S : Bar {
    fn g(&self) {
        self.f(self);
    }
}

fn main() {
    let s = S { x: 1 };
    s.g();
}

