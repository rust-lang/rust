// Test that we don't ICE for a typeck error that only shows up in dropck
// Version that uses a generic associated type
// Regression test for #91985

pub trait Trait1 {
    type Associated: Ord;
}

pub trait Trait2 {
    type Associated: Clone;
}

pub trait GatTrait {
    type Gat<T: Clone>;
}

pub struct GatStruct;

impl GatTrait for GatStruct {
    type Gat<T: Clone> = Box<T>;
}

pub struct OuterStruct<T1: Trait1, T2: Trait2> {
    inner: InnerStruct<T2, GatStruct>,
    t1: T1,
}

pub struct InnerStruct<T: Trait2, G: GatTrait> {
    pub gat: G::Gat<T::Associated>,
}

impl<T1, T2> OuterStruct<T1, T2>
where
    T1: Trait1,
    T2: Trait2<Associated = T1::Associated>,
{
    pub fn new() -> Self {
        //~^ ERROR the trait bound `<T1 as Trait1>::Associated: Clone` is not satisfied
        todo!()
    }
}

pub fn main() {}
