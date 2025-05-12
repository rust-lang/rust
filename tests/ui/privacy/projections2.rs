mod m {
    use super::*;
    struct Priv;
    pub type Leak = Priv; //~ WARN: `Priv` is more private than the item `Leak`

    trait Trait3 {
        type A<T: Trait>;
    }

    impl Trait3 for u8 {
        type A<T: Trait> = T::A<Leak>;
    }

    pub trait Trait4 {
        type A<T: Trait>;
    }

    impl Trait4 for u8 {
        type A<T: Trait> = <u8 as Trait3>::A<T>;
        //~^ ERROR: private associated type `Trait3::A` in public interface
        //~| ERROR: private trait `Trait3` in public interface
    }
}

pub trait Trait {
    type A<T>;
}

impl Trait for u8 {
    type A<T> = u8;
}
use m::*;

fn check4() -> <u8 as Trait4>::A<u8> {
    todo!()
}

fn main() {}
