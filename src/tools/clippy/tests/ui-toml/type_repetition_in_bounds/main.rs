#![allow(clippy::needless_maybe_sized)]
#![warn(clippy::type_repetition_in_bounds)]

fn f<T>()
where
    T: Copy + Clone + Sync + Send + ?Sized + Unpin,
    T: PartialEq,
{
}

fn f2<T>()
where
    T: Copy + Clone + Sync + Send + ?Sized,
    T: Unpin + PartialEq,
    //~^ type_repetition_in_bounds
{
}

fn main() {}
