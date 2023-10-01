#[rustfmt::skip]
pub struct Foo {
    /// 位	
    //~^ ERROR: using tabs in doc comments is not recommended
    //~| NOTE: `-D clippy::tabs-in-doc-comments` implied by `-D warnings`
    ///   ^ Do not remove this tab character.
    ///   It was required to trigger the ICE.
    pub bar: u8,
}

fn main() {}
