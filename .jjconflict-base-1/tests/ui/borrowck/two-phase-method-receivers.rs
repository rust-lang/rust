//@ run-pass

struct Foo<'a> {
    x: &'a i32
}

impl<'a> Foo<'a> {
    fn method(&mut self, _: &i32) {
    }
}

fn main() {
    let a = &mut Foo { x: &22 };
    Foo::method(a, a.x);
}
