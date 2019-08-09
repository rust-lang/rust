// run-pass
// pretty-expanded FIXME #23616

trait B {
    fn f(&self);
}

trait T : B {
}

struct A;

impl<U: T> B for U {
    fn f(&self) { }
}

impl T for A {
}

fn main() {
    let a = A;
    let br = &a as &dyn B;
    br.f();
}
