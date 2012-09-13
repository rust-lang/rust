use to_str::ToStr;

trait Foo {
    fn f(&self);
}

struct Bar {
    x: int
}

trait Baz {
    fn g(&self);
}

impl<T:Baz> T : Foo {
    fn f(&self) {
        self.g();
    }
}

impl Bar : Baz {
    fn g(&self) {
        io::println(self.x.to_str());
    }
}

fn main() {
    let y = Bar { x: 42 };
    y.f();
}