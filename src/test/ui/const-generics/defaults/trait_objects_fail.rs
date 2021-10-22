#![feature(const_generics_defaults)]

trait Trait<const N: u8 = 12> {
    fn uwu(&self) -> u8 {
        N
    }
}

impl Trait<2> for u32 {}

fn foo(arg: &dyn Trait) -> u8 {
    arg.uwu()
}

trait Traitor<const N: u8 = 1, const M: u8 = N> {
    fn owo(&self) -> u8 {
        M
    }
}

impl Traitor<2, 3> for bool { }

fn bar<const N: u8>(arg: &dyn Traitor<N>) -> u8 {
    arg.owo()
}

fn main() {
    foo(&10_u32);
    //~^ error: the trait bound `u32: Trait` is not satisfied
    bar(&true);
    //~^ error: the trait bound `bool: Traitor<{_: u8}, {_: u8}>` is not satisfied
}
