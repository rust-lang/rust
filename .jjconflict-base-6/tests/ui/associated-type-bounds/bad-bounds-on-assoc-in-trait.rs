//@ check-pass

use std::fmt::Debug;
use std::iter::Once;

trait Lam<Binder> {
    type App;
}

#[derive(Clone)]
struct L1;
impl<'a> Lam<&'a u8> for L1 {
    type App = u8;
}

#[derive(Clone)]
struct L2;
impl<'a, 'b> Lam<&'a &'b u8> for L2 {
    type App = u8;
}

trait Case1 {
    type C: Clone + Iterator<Item: Send + Iterator<Item: for<'a> Lam<&'a u8, App: Debug>> + Sync>;
}

pub struct S1;
impl Case1 for S1 {
    type C = Once<Once<L1>>;
}

fn assume_case1<T: Case1>() {
    fn assert_c<_1, _2, C>()
    where
        C: Clone + Iterator<Item = _2>,
        _2: Send + Iterator<Item = _1>,
        _1: for<'a> Lam<&'a u8>,
        for<'a> <_1 as Lam<&'a u8>>::App: Debug,
    {
    }
    assert_c::<_, _, T::C>();
}

fn main() {
    assume_case1::<S1>();
}
