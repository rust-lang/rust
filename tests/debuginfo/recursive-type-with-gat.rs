//@ compile-flags: -Cdebuginfo=2
#![allow(sized_hierarchy_migration)]
#![feature(sized_hierarchy)] // added to keep parameters unconstrained

pub trait Functor: std::marker::PointeeSized
{
    type With<T>: Functor;
}

pub struct IdFunctor<T>(T);
impl<T> Functor for IdFunctor<T> {
    type With<T2> = IdFunctor<T2>;
}

impl<T> Functor for Vec<T> {
    type With<T2> = Vec<T2> ;
}


pub struct Compose<F1, F2, T>(F1::With<F2::With<T>>)
where
    F1: Functor + std::marker::PointeeSized,
    F2: Functor + std::marker::PointeeSized;

impl<F1, F2, T> Functor for Compose<F1, F2, T>
where
    F1: Functor + std::marker::PointeeSized,
    F2: Functor + std::marker::PointeeSized,
{
    type With<T2> = F1::With<F2::With<T2>> ;
}

pub enum Value<F>
where
    F: Functor + std::marker::PointeeSized,
{
    SignedInt(*mut F::With<i64>),
    Array(*mut Value<Compose<F, Vec<()>, ()>>),

}

fn main() {
    let x: Value<IdFunctor<()>> = Value::SignedInt(&mut IdFunctor(1));
}
