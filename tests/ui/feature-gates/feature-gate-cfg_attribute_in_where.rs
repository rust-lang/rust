//@ revisions: a b

#![crate_type = "lib"]
use std::marker::PhantomData;

#[cfg(a)]
trait TraitA {}

#[cfg(b)]
trait TraitB {}

#[cfg(all())]
trait Defined {}

#[cfg(any())]
trait Undefined {}

trait A<T>
where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{
    type B<U>
    where
        #[cfg(a)] U: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(b)] U: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(any())] T: Undefined; //~ `#[cfg]` attribute in `where` clause is unstable

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(b)] U: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(any())] T: Undefined; //~ `#[cfg]` attribute in `where` clause is unstable
}

impl<T> A<T> for T
where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{
    type B<U> = () where
        #[cfg(a)] U: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(b)] U: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(any())] T: Undefined; //~ `#[cfg]` attribute in `where` clause is unstable

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(b)] U: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(any())] T: Undefined {} //~ `#[cfg]` attribute in `where` clause is unstable
}

struct C<T>
where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{
    _t: PhantomData<T>,
}

union D<T>
where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{

    _t: PhantomData<T>,
}

enum E<T>
where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{
    E(PhantomData<T>),
}

impl<T> C<T> where
    #[cfg(a)] T: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(b)] T: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
    #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
{
    fn new<U>() where
        #[cfg(a)] U: TraitA, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(b)] U: TraitB, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(all())] T: Defined, //~ `#[cfg]` attribute in `where` clause is unstable
        #[cfg(any())] T: Undefined, //~ `#[cfg]` attribute in `where` clause is unstable
    {}
}
