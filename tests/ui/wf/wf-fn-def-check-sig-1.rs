// Regression test for #84533.

use std::marker::PhantomData;

fn foo<'b, 'a>() -> PhantomData<&'b &'a ()> {
    PhantomData
}

fn extend_lifetime<'a, 'b, T: ?Sized>(x: &'a T) -> &'b T {
    let f = foo::<'b, 'a>;
    f.baz(x)
    //~^ ERROR lifetime may not live long enough
}

trait Foo<'a, 'b, T: ?Sized> {
    fn baz(self, s: &'a T) -> &'b T;
}
impl<'a, 'b, R, F, T: ?Sized> Foo<'a, 'b, T> for F
where
    F: Fn() -> R,
    R: ProofForConversion<'a, 'b, T>,
{
    fn baz(self, s: &'a T) -> &'b T {
        self().convert(s)
    }
}

trait ProofForConversion<'a, 'b, T: ?Sized> {
    fn convert(self, s: &'a T) -> &'b T;
}
impl<'a, 'b, T: ?Sized> ProofForConversion<'a, 'b, T> for PhantomData<&'b &'a ()> {
    fn convert(self, s: &'a T) -> &'b T {
        s
    }
}

fn main() {
    let d;
    {
        let x = "Hello World".to_string();
        d = extend_lifetime(&x);
    }
    println!("{}", d);
}
