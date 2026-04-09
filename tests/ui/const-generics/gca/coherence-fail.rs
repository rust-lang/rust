#![feature(generic_const_items, min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

// computing the same value with different constant items but same generic arguments should fail
trait Trait1 {}
type const FOO<const N: usize>: usize = const { N + 1 };
type const BAR<const N: usize>: usize = const { N + 1 };
impl Trait1 for [(); FOO::<1>] {}
impl Trait1 for [(); BAR::<1>] {}
//~^ ERROR conflicting implementations of trait `Trait1` for type `[(); 2]`

// computing the same value with the same constant item but different generic arguments should fail
type const DIV2<const N: usize>: usize = const { N / 2 };
trait Trait2 {}
impl Trait2 for [(); DIV2::<2>] {}
impl Trait2 for [(); DIV2::<3>] {}
//~^ ERROR conflicting implementations of trait `Trait2` for type `[(); 1]`

// computing the same value with different constant items and different generic arguments should
// fail
trait Trait3 {}
type const ADD1<const N: usize>: usize = const { N + 1 };
type const SUB1<const N: usize>: usize = const { N - 1 };
impl Trait3 for [(); ADD1::<1>] {}
impl Trait3 for [(); SUB1::<3>] {}
//~^ ERROR conflicting implementations of trait `Trait3` for type `[(); 2]`

fn main() {}
