//@ check-pass

#![feature(non_lifetime_binders)]

pub fn foo()
where
    for<V> V: Sized,
{
    bar();
}

pub fn bar()
where
    for<V> V: Sized,
{
}

pub fn main() {}
