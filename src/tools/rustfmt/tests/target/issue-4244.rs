pub struct SS {}

pub type A /* A Comment */ = SS;

pub type B // Comment
    // B
    = SS;

pub type C
    /* Comment C */
    = SS;

pub trait D<T> {
    type E /* Comment E */ = SS;
}

type F<'a: 'static, T: Ord + 'static>: Eq + PartialEq
where
    T: 'static + Copy, /* x */
= Vec<u8>;
