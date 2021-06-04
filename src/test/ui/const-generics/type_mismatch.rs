fn foo<const N: usize>() -> [u8; N] {
    bar::<N>() //~ ERROR mismatched types
}

fn bar<const N: u8>() -> [u8; N] {}
//~^ ERROR mismatched types
//~| ERROR mismatched types

fn main() {}
