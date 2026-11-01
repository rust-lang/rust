mod m {
    struct Priv;
    pub type Leak = Priv; //~ WARN: `Priv` is more private than the item `Leak`
}

trait Trait {
    type A<T>;
}

impl Trait for u8 {
    type A<T> = u8;
}

fn check() -> <u8 as Trait>::A<m::Leak> {
    //~^ ERROR: `Priv` is private
    0
}

trait Trait2 {
    type A<T>;
}

impl Trait2 for u8 {
    type A<T> = m::Leak;
    //~^ ERROR: `Priv` is private
    //~| ERROR: private type `Priv` in public interface
}

fn check2() -> <u8 as Trait2>::A<u32> {
    //~^ ERROR: `Priv` is private
    todo!()
}

trait Trait3 {
    type A<T: Trait>;
}

impl Trait3 for u8 {
    type A<T: Trait> = T::A<m::Leak>;
    //~^ ERROR: `Priv` is private
    //~| ERROR: private type `Priv` in public interface
}

fn check3() -> <u8 as Trait3>::A<u8> {
    todo!()
}

trait Trait4 {
    type A<T: Trait3>;
}

impl Trait4 for u8 {
    type A<T: Trait3> = T::A<u8>;
}

fn check4() -> <u8 as Trait4>::A<u8> {
    todo!()
}

fn main() {}
