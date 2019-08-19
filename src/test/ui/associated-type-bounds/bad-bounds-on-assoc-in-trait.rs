// compile-fail
// ignore-tidy-linelength

// NOTE: rustc cannot currently handle bounds of the form `for<'a> <Foo as Bar<'a>>::Assoc: Baz`.
// This should hopefully be fixed with Chalk.

#![feature(associated_type_bounds)]

use std::fmt::Debug;
use std::iter::Once;

trait Lam<Binder> { type App; }

#[derive(Clone)]
struct L1;
impl<'a> Lam<&'a u8> for L1 { type App = u8; }

#[derive(Clone)]
struct L2;
impl<'a, 'b> Lam<&'a &'b u8> for L2 { type App = u8; }

trait Case1 {
    type C: Clone + Iterator<Item:
        Send + Iterator<Item:
            for<'a> Lam<&'a u8, App:
                Debug
            >
        > + Sync>;
}

pub struct S1;
impl Case1 for S1 {
//~^ ERROR `<L1 as Lam<&'a u8>>::App` doesn't implement `std::fmt::Debug` [E0277]
    type C = Once<Once<L1>>;
}

fn assume_case1<T: Case1>() {
//~^ ERROR `<_ as Lam<&'a u8>>::App` doesn't implement `std::fmt::Debug` [E0277]
//~| ERROR `<<T as Case1>::C as std::iter::Iterator>::Item` is not an iterator [E0277]
//~| ERROR `<<T as Case1>::C as std::iter::Iterator>::Item` cannot be sent between threads safely [E0277]
//~| ERROR `<<T as Case1>::C as std::iter::Iterator>::Item` cannot be shared between threads safely [E0277]
    fn assert_a<_0, A>() where A: Iterator<Item = _0>, _0: Debug {}
    assert_a::<_, T::A>();

    fn assert_b<_0, B>() where B: Iterator<Item = _0>, _0: 'static {}
    assert_b::<_, T::B>();

    fn assert_c<_0, _1, _2, C>()
    where
        C: Clone + Iterator<Item = _2>,
        _2: Send + Iterator<Item = _1>,
        _1: for<'a> Lam<&'a u8, App = _0>,
        _0: Debug,
    {}
    assert_c::<_, _, _, T::C>();
}

fn main() {
    assume_case1(S1);
}
