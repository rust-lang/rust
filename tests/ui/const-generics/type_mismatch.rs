fn foo<const N: usize>() -> [u8; N] {
    bar::<N>()
}

fn bar<const N: u8>() -> [u8; N] {}
//~^ ERROR the constant `N` is not of type `usize`

fn main() {}
