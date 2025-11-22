#![feature(arbitrary_self_types)]
#![feature(unsize)]
#![feature(dispatch_from_dyn)]

use std::marker::{PhantomData, Unsize};
use std::ops::{Deref, DispatchFromDyn, Receiver};

struct IsSendToken<T: ?Sized>(PhantomData<fn(T) -> T>);

struct Foo<'a, U: ?Sized> {
    token: IsSendToken<U>,
    ptr: &'a U,
}

impl<'a, T, U> DispatchFromDyn<Foo<'a, U>> for Foo<'a, T>
//~^ ERROR implementing `DispatchFromDyn` does not allow multiple fields to be coerced
where
    T: Unsize<U> + ?Sized,
    U: ?Sized,
{
}

trait Bar {
    fn f(self: Foo<'_, Self>);
}

impl<U: ?Sized> Deref for Foo<'_, U> {
    type Target = U;
    fn deref(&self) -> &U {
        self.ptr
    }
}
impl<U: ?Sized> Receiver for Foo<'_, U> {
    type Target = U;
}

fn main() {}
