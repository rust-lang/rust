// rustfmt-inline_attribute_width: 40

#![crate_type = "lib"]
#![feature(cfg_attribute_in_where)]
use std::marker::PhantomData;

#[cfg(a)]
trait TraitA {}

#[cfg(b)]
trait TraitB {}

trait A<T>
where
    #[cfg = a_very_long_attribute_name]
    T: TraitA,
    #[cfg = another_very_long_attribute_name]
    T: TraitB,
{
    type B<U>
    where
        #[cfg = a]
        // line comment after the attribute
        U: TraitA,
        #[cfg = b]
        /* block comment after the attribute */
        U: TraitB,
        #[cfg = a] // short
        U: TraitA,
        #[cfg = b] /* short */ U: TraitB;

    fn foo<U>(&self)
    where
        /// line doc comment before the attribute
        U: TraitA,
        /** line doc block comment before the attribute */
        U: TraitB;
}

impl<T> A<T> for T
where
    #[doc = "line doc before the attribute"]
    T: TraitA,
    /** short doc */
    T: TraitB,
{
    type B<U>
        = ()
    where
        #[doc = "short"] U: TraitA,
        #[doc = "short"]
        #[cfg = a]
        U: TraitB;

    fn foo<U>(&self)
    where
        #[cfg = a]
        #[cfg = b]
        U: TraitA,
        /// line doc
        #[cfg = c]
        U: TraitB,
    {
    }
}

struct C<T>
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
{
    _t: PhantomData<T>,
}

union D<T>
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
{
    _t: PhantomData<T>,
}

enum E<T>
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
{
    E(PhantomData<T>),
}

#[allow(type_alias_bounds)]
type F<T>
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
= T;

impl<T> C<T>
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
{
    fn new<U>()
    where
        #[cfg = a] U: TraitA,
        #[cfg = b] U: TraitB,
    {
    }
}

fn foo<T>()
where
    #[cfg = a] T: TraitA,
    #[cfg = b] T: TraitB,
{
}
