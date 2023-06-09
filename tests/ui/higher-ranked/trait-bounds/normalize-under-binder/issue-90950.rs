// check-fail
// known-bug: #90950

trait Yokeable<'a>: 'static {
    type Output: 'a;
}


trait IsCovariant<'a> {}

struct Yoke<Y: for<'a> Yokeable<'a>> {
    data: Y,
}


// impl<Y: for<'a> Yokeable<'a>> Yoke<Y> {
//     fn project<Y2: for<'a> Yokeable<'a>>(
//         &self,
//         f: for<'a> fn(<Y as Yokeable<'a>>::Output, &'a (),
//     ) -> <Y2 as Yokeable<'a>>::Output) -> Yoke<Y2> {
//         unimplemented!()
//     }
// }

fn upcast<Y>(x: Yoke<Y>) -> Yoke<Box<dyn IsCovariant<'static> + 'static>> where
    Y: for<'a> Yokeable<'a>,
    for<'a> <Y as Yokeable<'a>>::Output: IsCovariant<'a>
    {
    // x.project(|data, _| {
    //     Box::new(data)
    // })
    unimplemented!()
}


impl<'a> Yokeable<'a> for Box<dyn IsCovariant<'static> + 'static> {
    type Output = Box<dyn IsCovariant<'a> + 'a>;
}

// this impl is mostly an example and unnecessary for the pure repro
use std::borrow::*;
impl<'a, T: ToOwned + ?Sized> Yokeable<'a> for Cow<'static, T> {
    type Output = Cow<'a, T>;
}
impl<'a, T: ToOwned + ?Sized> IsCovariant<'a> for Cow<'a, T> {}



fn upcast_yoke(y: Yoke<Cow<'static, str>>) -> Yoke<Box<dyn IsCovariant<'static> + 'static>> {
    upcast(y)
}

fn main() {}
