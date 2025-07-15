//@ edition: 2021
// https://github.com/rust-lang/rust/issues/112713

pub fn foo() -> u32 {
    use sub::*;
    C
    //~^ ERROR `C` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

mod sub {
    mod mod1 { pub const C: u32 = 1; }
    mod mod2 { pub const C: u32 = 2; }

    pub use mod1::*;
    pub use mod2::*;
}

fn main() {}
