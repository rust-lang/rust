struct Uwu<const N: u32 = 1, const M: u32 = N>;

trait Trait {}
impl<const N: u32> Trait for Uwu<N> {}

fn rawr() -> impl Trait {
    //~^ error: the trait bound `Uwu<10, 12>: Trait` is not satisfied
    Uwu::<10, 12>
}

trait Traitor<const N: u8 = 1, const M: u8 = N> {}

impl<const N: u8> Traitor<N, 2> for u32 {}
impl Traitor<1, 2> for u64 {}

fn uwu<const N: u8>() -> impl Traitor<N> {
    //~^ error: the trait bound `u32: Traitor<N>` is not satisfied
    1_u32
}

fn owo() -> impl Traitor {
    //~^ error: the trait bound `u64: Traitor` is not satisfied
    1_u64
}

fn main() {
    rawr();
    uwu(); //~ ERROR: type annotations needed
    owo();
}
