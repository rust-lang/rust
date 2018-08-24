// ignore-compare-mode-nll

#![allow(unused_variables)]

trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

fn method1<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _c: <T as Trait<'a>>::Type = a;
}

fn method2<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _c: <T as Trait<'b>>::Type = a; //~ ERROR E0623
}

fn method3<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _c: <T as Trait<'a>>::Type = b; //~ ERROR E0623
}

fn method4<'a,'b,T>(x: &'a T, y: &'b T)
    where T : for<'z> Trait<'z>, 'a : 'b
{
    // Note that &'static T <: &'a T.
    let a: <T as Trait<'a>>::Type = loop { };
    let b: <T as Trait<'b>>::Type = loop { };
    let _c: <T as Trait<'b>>::Type = b;
}

fn main() { }
