trait Foo {
    fn f(&self);
}

struct S {
    x: int
}

impl S : Foo {
    fn f(&self) {
        assert self.x == 3;
    }
}

fn main() {
    let x = @S { x: 3 };
    let y = x as @Foo;
    y.f();
}


