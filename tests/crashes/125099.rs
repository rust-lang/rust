//@ known-bug: rust-lang/rust#125099

pub trait ContFn<T>: Fn(T) -> Self::Future {
    type Future;
}
impl<T, F> ContFn<T> for F
where
    F: Fn(T),
{
    type Future = ();
}

pub trait SeqHandler {
    type Requires;
    fn process<F: ContFn<Self::Requires>>() -> impl Sized;
}

pub struct ConvertToU64;
impl SeqHandler for ConvertToU64 {
    type Requires = u64;
    fn process<F: ContFn<Self::Requires>>() -> impl Sized {}
}

fn main() {}
