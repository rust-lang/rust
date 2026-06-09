#![allow(unused_variables)]

fn make_any<T>() -> T {  loop {} }

trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

fn method1<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = make_any();
    let b: <T as Trait<'b>>::Type = make_any();
    let _c: <T as Trait<'a>>::Type = a;
}

fn method2<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = make_any();
    //~^ ERROR lifetime may not live long enough
    let b: <T as Trait<'b>>::Type = make_any();
    let _c: <T as Trait<'b>>::Type = a;
}

fn method3<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = make_any();
    let b: <T as Trait<'b>>::Type = make_any();
    let _c: <T as Trait<'a>>::Type = b;
    //~^ ERROR lifetime may not live long enough
}

fn method4<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = make_any();
    let b: <T as Trait<'b>>::Type = make_any();
    let _c: <T as Trait<'b>>::Type = b;
}

fn main() { }
