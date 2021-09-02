#![feature(generic_const_exprs, array_map)]
#![allow(incomplete_features)]

pub struct ConstCheck<const CHECK: bool>;

pub trait True {}
impl True for ConstCheck<true> {}

pub trait OrdesDec {
    type Newlen;
    type Output;

    fn pop(self) -> (Self::Newlen, Self::Output);
}

impl<T, const N: usize> OrdesDec for [T; N]
where
    ConstCheck<{N > 1}>: True,
    [T; N - 1]: Sized,
{
    type Newlen = [T; N - 1];
    type Output = T;

    fn pop(self) -> (Self::Newlen, Self::Output) {
        let mut iter = IntoIter::new(self);
        //~^ ERROR: failed to resolve: use of undeclared type `IntoIter`
        let end = iter.next_back().unwrap();
        let new = [(); N - 1].map(move |()| iter.next().unwrap());
        (new, end)
    }
}

fn main() {}
