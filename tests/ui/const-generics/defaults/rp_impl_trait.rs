//@ run-pass
struct Uwu<const N: u32 = 1, const M: u32 = N>;

trait Trait {}
impl<const N: u32> Trait for Uwu<N> {}

fn rawr<const N: u32>() -> impl Trait {
    Uwu::<N>
}

trait Traitor<const N: u8 = 1, const M: u8 = N> { }

impl<const N: u8> Traitor<N> for u32 {}
impl Traitor<1, 1> for u64 {}

fn uwu<const N: u8>() -> impl Traitor<N> {
    1_u32
}

fn owo() -> impl Traitor {
    1_u64
}

fn main() {
    rawr::<3>();
    rawr::<7>();
    uwu::<{ u8::MAX }>();
    owo();
}
