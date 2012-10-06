trait Foo {
    fn f() -> int;
}

struct A {
    x: int
}

impl A : Foo {
    fn f() -> int {
        io::println(~"Today's number is " + self.x.to_str());
        return self.x;
    }
}

fn main() {
    let a = A { x: 3 };
    let b = (&a) as &Foo;
    assert b.f() == 3;
}

