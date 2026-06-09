//@ check-pass
//@ compile-flags: --crate-type=lib

#![allow(dead_code)]

use std::marker::PhantomData;

pub struct XImpl<T, E, F2, F1>
where
    F2: Fn(E),
{
    f1: F1,
    f2: F2,
    _ghost: PhantomData<(T, E)>,
}

pub trait X<T>: Sized {
    type F1;
    type F2: Fn(Self::E);
    type E;

    fn and<NewF1, NewF1Generator>(self, f: NewF1Generator) -> XImpl<T, Self::E, Self::F2, NewF1>
    where
        NewF1Generator: FnOnce(Self::F1) -> NewF1;
}

impl<T, E, F2, F1> X<T> for XImpl<T, E, F2, F1>
where
    F2: Fn(E),
{
    type E = E;
    type F2 = F2;
    type F1 = F1;

    fn and<NewF1, NewF1Generator>(self, f: NewF1Generator) -> XImpl<T, E, F2, NewF1>
    where
        NewF1Generator: FnOnce(F1) -> NewF1,
    {
        XImpl {
            f1: f(self.f1),
            f2: self.f2,
            _ghost: PhantomData,
        }
    }
}

fn f() -> impl X<(), E = ()> {
    XImpl {
        f1: || (),
        f2: |()| (),
        _ghost: PhantomData,
    }
}

fn f2() -> impl X<(), E = ()> {
    f().and(|rb| rb)
}
