//@ edition: 2021
// https://github.com/rust-lang/rust/issues/112713

pub fn foo() -> u32 {
    use sub::*;
    C
    //~^ ERROR `C` is ambiguous
}

mod sub {
    mod mod1 { pub const C: u32 = 1; }
    mod mod2 { pub const C: u32 = 2; }

    pub use mod1::*;
    pub use mod2::*;
}

fn main() {}
