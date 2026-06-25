//@ compile-flags: -Znext-solver
//@ check-pass

fn foo<const N: usize>() -> impl Iterator<Item = [u8; N]> {
    std::iter::empty()
}

fn main() {}
