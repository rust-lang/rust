pub struct Foo<T>(T);

impl<T> Foo<T> {
    pub fn method(&self) {}
}

fn do_stuff<'a, T>(foo: Foo<T>)  {
    foo.method();
}

fn main() {}