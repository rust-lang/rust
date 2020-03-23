trait Foo: Sized {
    fn foo(self);
}

fn foo<'a,'b,T>(x: &'a T, y: &'b T)
    where &'a T : Foo, //~ ERROR type annotations needed
          &'b T : Foo
{
    x.foo();
    y.foo();
}

fn main() { }
