//@ revisions: a b
//@ check-pass

#![crate_type = "lib"]
#![feature(cfg_attribute_in_where)]
use std::marker::PhantomData;

#[cfg(a)]
trait TraitA {}

#[cfg(b)]
trait TraitB {}

trait A<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
    type B<U>
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB;

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB;
}

impl<T> A<T> for T
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
    type B<U> = () where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB;

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB, {}
}

struct C<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
    _t: PhantomData<T>,
}

union D<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{

    _t: PhantomData<T>,
}

enum E<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
    E(PhantomData<T>),
}

#[allow(type_alias_bounds)]
type F<T>
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB = T;

impl<T> C<T> where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
    fn new<U>() where
        #[cfg(a)] U: TraitA,
        #[cfg(b)] U: TraitB,
    {}
}

fn foo<T>()
where
    #[cfg(a)] T: TraitA,
    #[cfg(b)] T: TraitB,
{
}
