trait Foo {
    fn a() -> int;
    fn b() -> int {
        self.a() + 2
    }
}

impl int: Foo {
    fn a() -> int {
        3
    }
}

fn main() {
    assert(3.b() == 5);
}
