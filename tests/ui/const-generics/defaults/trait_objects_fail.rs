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

impl Traitor<2, 3> for bool {}

fn bar<const N: u8>(arg: &dyn Traitor<N>) -> u8 {
    arg.owo()
}

fn main() {
    foo(&10_u32);
    //~^ ERROR trait `Trait` is not implemented for `u32`
    bar(&true);
    //~^ ERROR trait `Traitor<_>` is not implemented for `bool`
}
