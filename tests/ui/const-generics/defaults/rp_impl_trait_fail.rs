struct Uwu<const N: u32 = 1, const M: u32 = N>;

trait Trait {}
impl<const N: u32> Trait for Uwu<N> {}

fn rawr() -> impl Trait {
    //~^ ERROR trait `Trait` is not implemented for `Uwu<10, 12>`
    Uwu::<10, 12>
}

trait Traitor<const N: u8 = 1, const M: u8 = N> {}

impl<const N: u8> Traitor<N, 2> for u32 {}
impl Traitor<1, 2> for u64 {}

fn uwu<const N: u8>() -> impl Traitor<N> {
    //~^ ERROR trait `Traitor<N>` is not implemented for `u32`
    1_u32
}

fn owo() -> impl Traitor {
    //~^ ERROR trait `Traitor` is not implemented for `u64`
    1_u64
}

fn main() {
    rawr();
    uwu(); //~ ERROR: type annotations needed
    owo();
}
