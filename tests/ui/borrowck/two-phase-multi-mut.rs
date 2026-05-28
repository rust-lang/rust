struct Foo {
}

impl Foo {
    fn method(&mut self, foo: &mut Foo) {
    }
}

fn main() {
    let mut foo = Foo { };
    foo.method(&mut foo);
    //~^ ERROR cannot borrow `foo` as mutable more than once at a time
    //~^^ ERROR cannot borrow `foo` as mutable more than once at a time
}
