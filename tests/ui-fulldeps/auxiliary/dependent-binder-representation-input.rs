pub trait Family<const N: usize> {
    type Item;
}

impl<const N: usize> Family<N> for () {
    type Item = [u8; N];
}

impl<const N: usize> Family<N> for bool {
    type Item = [bool; N];
}

pub trait Other {
    type Item;
}

pub trait Borrowing<'a> {
    const VALUE: usize;
}

impl<'a> Borrowing<'a> for () {
    const VALUE: usize = 0;
}

pub trait ReturnType {
    fn method<'a>() -> impl Sized + 'a;
}

fn main() {}
