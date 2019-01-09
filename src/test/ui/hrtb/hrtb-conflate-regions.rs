// Test that an impl with only one bound region `'a` cannot be used to
// satisfy a constraint where there are two bound regions.

trait Foo<X> {
    fn foo(&self, x: X) { }
}

fn want_foo2<T>()
    where T : for<'a,'b> Foo<(&'a isize, &'b isize)>
{
}

fn want_foo1<T>()
    where T : for<'z> Foo<(&'z isize, &'z isize)>
{
}

///////////////////////////////////////////////////////////////////////////
// Expressed as a where clause

struct SomeStruct;

impl<'a> Foo<(&'a isize, &'a isize)> for SomeStruct
{
}

fn a() { want_foo1::<SomeStruct>(); } // OK -- foo wants just one region
fn b() { want_foo2::<SomeStruct>(); } //~ ERROR

fn main() { }
