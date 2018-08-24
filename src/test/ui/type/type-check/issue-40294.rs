trait Foo: Sized {
    fn foo(self);
}

fn foo<'a,'b,T>(x: &'a T, y: &'b T) //~ ERROR type annotations required
    where &'a T : Foo,
          &'b T : Foo
{
    x.foo();
    y.foo();
}

fn main() { }
