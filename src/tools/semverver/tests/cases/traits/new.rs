pub trait Abc {
    type A;
    fn test(&self) -> u8;
    fn test3(&self) -> u8;
    fn test4() -> u8 {
        0
    }
    fn test6() -> u8 {
        0
    }
    fn test7() -> u16;
    fn test8(_: &Self) -> u8;
    fn test9(&self) -> u8;
}

pub trait Bcd<A> {}

pub trait Cde {}

pub trait Def<A, B> {
    // The method is not broken - the impls are, but calls should work as expected, as
    // long as a proper impl is presented. Maybe this will need some more careful handling.
    fn def(&self, a: B) -> bool;
}

pub trait Efg<A> {
    fn efg(&self, a: A) -> bool;
}

mod fgh {
    pub trait Fgh {
        fn fgh(&self) -> u8;
    }
}

pub trait Ghi { }

pub trait Hij {
    type A;
}

pub trait Klm : Clone { }

pub trait Nop { }

pub trait Qrs<A: Clone> { }

pub trait Tuv<A> { }
