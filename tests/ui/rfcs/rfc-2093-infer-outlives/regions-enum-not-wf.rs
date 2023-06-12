// Various examples of structs whose fields are not well-formed.

#![allow(dead_code)]

trait Dummy<'a> {
    type Out;
}
impl<'a, T> Dummy<'a> for T
where
    T: 'a,
{
    type Out = ();
}
type RequireOutlives<'a, T> = <T as Dummy<'a>>::Out;

enum Ref1<'a, T> {
    Ref1Variant1(RequireOutlives<'a, T>), //~ ERROR the parameter type `T` may not live long enough
}

enum Ref2<'a, T> {
    Ref2Variant1,
    Ref2Variant2(isize, RequireOutlives<'a, T>), //~ ERROR the parameter type `T` may not live long enough
}

enum RefOk<'a, T: 'a> {
    RefOkVariant1(&'a T),
}

// This is now well formed. RFC 2093
enum RefIndirect<'a, T> {
    RefIndirectVariant1(isize, RefOk<'a, T>),
}

enum RefDouble<'a, 'b, T> {
    RefDoubleVariant1(&'a RequireOutlives<'b, T>),
    //~^ the parameter type `T` may not live long enough [E0309]
}

fn main() {}
