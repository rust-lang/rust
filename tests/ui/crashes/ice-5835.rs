#[rustfmt::skip]
pub struct Foo {
    /// 位	
    ///   ^ Do not remove this tab character.
    ///   It was required to trigger the ICE.
    pub bar: u8,
}

fn main() {}
