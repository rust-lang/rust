use std::ops::AsyncFnMut;

pub trait Main {
    type Item;

    type Out0: Support<Item = ()>;
    type Out1: Support<Item = Self::Item>;
    type Out2<T>: Support<Item = T>;
    type Out3: Support<Produce<()> = bool>;
    type Out4<T>: Support<Produce<T> = T>;
    type Out5: Support<Output<'static> = &'static ()>;
    type Out6: for<'a> Support<Output<'a> = &'a ()>;
    type Out7: Support<Item = String, Produce<i32> = u32> + Unrelated;
    type Out8: Unrelated + Protocol<i16, Q1 = u128, Q0 = ()>;
    type Out9: FnMut(i32) -> bool + Clone;
    type Out10<'q>: Support<Output<'q> = ()>;
    type Out11: for<'r, 's> Helper<A<'s> = &'s (), B<'r> = ()>;
    type Out12: for<'w> Helper<B<'w> = std::borrow::Cow<'w, str>, A<'w> = bool>;
    type Out13: for<'fst, 'snd> Aid<'snd, Result<'fst> = &'fst mut str>;
    type Out14<P: Copy + Eq, Q: ?Sized>;
    type Out15: AsyncFnMut(i32) -> bool;

    fn make<F>(_: F, _: impl FnMut(&str) -> bool)
    where
        F: FnOnce(u32) -> String,
        Self::Out2<()>: Protocol<u8, Q0 = Self::Item, Q1 = ()>;
}

pub trait Support {
    type Item;
    type Output<'a>;
    type Produce<T>;
}

pub trait Protocol<K> {
    type Q0;
    type Q1;
}

pub trait Unrelated {}

pub trait Helper {
    type A<'q>;
    type B<'q>;
}

pub trait Aid<'src> {
    type Result<'inter: 'src>;
}

pub trait Implementee {
    type Alias<T: Eq>
    where
        String: From<T>;
}

pub struct Implementor;

impl Implementee for Implementor {
    type Alias<T: Eq> = T
    where
        String: From<T>,
        // We will check that this bound doesn't get turned into an item bound since
        // associated types in impls are not allowed to have any.
        Self::Alias<T>: From<Self::Alias<T>>;
}
