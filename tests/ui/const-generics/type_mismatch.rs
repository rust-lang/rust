fn foo<const N: usize>() -> [u8; N] {
    bar::<N>() //~ ERROR mismatched types
    //~^ ERROR the constant `N` is not of type `u8`
}

fn bar<const N: u8>() -> [u8; N] {}
//~^ ERROR mismatched types
//~| ERROR mismatched types

fn main() {}
