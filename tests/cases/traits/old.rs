pub trait Abc {
    type A;
    fn test(&self) -> u8;
    fn test2(&self) -> u8;
    fn test4() -> u8 {
        0
    }
    fn test5() -> u8 {
        0
    }
    fn test7() -> u8;
    fn test8(&self) -> u8;
    fn test9(_: &Self) -> u8;
}

pub trait Bcd {}

pub trait Cde<A> {}

pub trait Def<A> {
    fn def(&self, a: A) -> bool;
}

pub trait Efg<A, B> {
    fn efg(&self, a: B) -> bool;
}

mod fgh {
    pub trait Fgh {
        fn fgh(&self) -> bool;
    }
}

pub trait Ghi {
    type A;
}

pub trait Hij { }

pub trait Klm { }

pub trait Nop : Clone { }

pub trait Qrs<A> { }

pub trait Tuv<A: Clone> { }
