struct Uwu<const N: u32 = 1, const M: u32 = N>;

trait Trait {}
impl<const N: u32> Trait for Uwu<N> {}

fn rawr() -> impl Trait {
    Uwu::<10, 12>
    //~^ error: the trait bound `Uwu<10_u32, 12_u32>: Trait` is not satisfied
}

trait Traitor<const N: u8 = 1, const M: u8 = N> { }

impl<const N: u8> Traitor<N, 2> for u32 {}
impl Traitor<1, 2> for u64 {}


fn uwu<const N: u8>() -> impl Traitor<N> {
    1_u32
    //~^ error: the trait bound `u32: Traitor<N, N>` is not satisfied
}

fn owo() -> impl Traitor {
    1_u64
    //~^ error: the trait bound `u64: Traitor<1_u8, 1_u8>` is not satisfied
}

fn main() {
    rawr();
    uwu();
    owo();
}
