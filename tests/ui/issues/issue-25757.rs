//@ run-pass
struct Foo {
    a: u32
}

impl Foo {
    fn x(&mut self) {
        self.a = 5;
    }
}

const FUNC: &'static dyn Fn(&mut Foo) -> () = &Foo::x;

fn main() {
    let mut foo = Foo { a: 137 };
    FUNC(&mut foo);
    assert_eq!(foo.a, 5);
}
