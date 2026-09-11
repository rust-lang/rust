#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

trait Other<U: ?Sized> {}

trait Trait<U>
where
    for<T> U: Other<T>,
{
}
