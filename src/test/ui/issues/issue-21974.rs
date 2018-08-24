// Test that (for now) we report an ambiguity error here, because
// specific trait relationships are ignored for the purposes of trait
// matching. This behavior should likely be improved such that this
// test passes. See #21974 for more details.

trait Foo {
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
