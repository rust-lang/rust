// Various examples of structs whose fields are not well-formed.

#![allow(dead_code)]

enum Ref1<'a, T> {
    Ref1Variant1(&'a T) //~ ERROR the parameter type `T` may not live long enough
}

enum Ref2<'a, T> {
    Ref2Variant1,
    Ref2Variant2(isize, &'a T), //~ ERROR the parameter type `T` may not live long enough
}

enum RefOk<'a, T:'a> {
    RefOkVariant1(&'a T)
}

enum RefIndirect<'a, T> {
    RefIndirectVariant1(isize, RefOk<'a,T>)
        //~^ ERROR the parameter type `T` may not live long enough
}

enum RefDouble<'a, 'b, T> {
    RefDoubleVariant1(&'a &'b T)
        //~^ ERROR reference has a longer lifetime than the data
}

fn main() { }
