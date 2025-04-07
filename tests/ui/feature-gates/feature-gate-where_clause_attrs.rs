//@ revisions: a b

#![crate_type = "lib"]
use std::marker::PhantomData;

#[cfg(a)]
trait TraitA {}

#[cfg(b)]
trait TraitB {}

#[cfg_attr(a, cfg(a))]
trait TraitAA {}

#[cfg_attr(b, cfg(b))]
trait TraitBB {}

#[cfg(all())]
trait TraitAll {}

#[cfg(any())]
trait TraitAny {}

#[cfg_attr(all(), cfg(all()))]
trait TraitAllAll {}

#[cfg_attr(any(), cfg(any()))]
trait TraitAnyAny {}


trait A<T>
where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{
    type B<U>
    where
        #[cfg(a)] U: TraitA, //~ ERROR attributes in `where` clause are unstable
        #[cfg(b)] U: TraitB, //~ ERROR attributes in `where` clause are unstable
        #[cfg(all())] U: TraitAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg(any())] U: TraitAny, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(a, cfg(a))] U: TraitAA, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(b, cfg(b))] U: TraitBB, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(all(), cfg(all()))] U: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(any(), cfg(any()))] U: TraitAnyAny; //~ ERROR attributes in `where` clause are unstable

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA, //~ ERROR attributes in `where` clause are unstable
        #[cfg(b)] U: TraitB, //~ ERROR attributes in `where` clause are unstable
        #[cfg(all())] U: TraitAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg(any())] U: TraitAny, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(a, cfg(a))] U: TraitAA, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(b, cfg(b))] U: TraitBB, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(all(), cfg(all()))] U: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(any(), cfg(any()))] U: TraitAnyAny; //~ ERROR attributes in `where` clause are unstable
}

impl<T> A<T> for T
where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{
    type B<U> = () where
        #[cfg(a)] U: TraitA, //~ ERROR attributes in `where` clause are unstable
        #[cfg(b)] U: TraitB, //~ ERROR attributes in `where` clause are unstable
        #[cfg(all())] U: TraitAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg(any())] U: TraitAny, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(a, cfg(a))] U: TraitAA, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(b, cfg(b))] U: TraitBB, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(all(), cfg(all()))] U: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(any(), cfg(any()))] U: TraitAnyAny; //~ ERROR attributes in `where` clause are unstable

    fn foo<U>(&self)
    where
        #[cfg(a)] U: TraitA, //~ ERROR attributes in `where` clause are unstable
        #[cfg(b)] U: TraitB, //~ ERROR attributes in `where` clause are unstable
        #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(a, cfg(a))] U: TraitAA, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(b, cfg(b))] U: TraitBB, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
    {}
}

struct C<T>
where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{
    _t: PhantomData<T>,
}

union D<T>
where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{

    _t: PhantomData<T>,
}

enum E<T>
where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{
    E(PhantomData<T>),
}

impl<T> C<T> where
    #[cfg(a)] T: TraitA, //~ ERROR attributes in `where` clause are unstable
    #[cfg(b)] T: TraitB, //~ ERROR attributes in `where` clause are unstable
    #[cfg(all())] T: TraitAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg(any())] T: TraitAny, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(a, cfg(a))] T: TraitAA, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(b, cfg(b))] T: TraitBB, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(all(), cfg(all()))] T: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
    #[cfg_attr(any(), cfg(any()))] T: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
{
    fn new<U>() where
        #[cfg(a)] U: TraitA, //~ ERROR attributes in `where` clause are unstable
        #[cfg(b)] U: TraitB, //~ ERROR attributes in `where` clause are unstable
        #[cfg(all())] U: TraitAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg(any())] U: TraitAny, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(a, cfg(a))] U: TraitAA, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(b, cfg(b))] U: TraitBB, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(all(), cfg(all()))] U: TraitAllAll, //~ ERROR attributes in `where` clause are unstable
        #[cfg_attr(any(), cfg(any()))] U: TraitAnyAny, //~ ERROR attributes in `where` clause are unstable
    {}
}
