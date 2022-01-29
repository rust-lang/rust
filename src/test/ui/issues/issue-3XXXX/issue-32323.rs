pub trait Tr<'a> {
    type Out;
}

pub fn f<'a, T: Tr<'a>>() -> <T as Tr<'a>>::Out {}
//~^ ERROR mismatched types

pub fn main() {}
