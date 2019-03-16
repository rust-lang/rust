// compile-fail

#![feature(associated_type_bounds)]

use std::fmt::Debug;

trait Lam<Binder> { type App; }

fn nested_bounds<_0, _1, _2, D>()
where
    D: Clone + Iterator<Item: Send + for<'a> Iterator<Item: for<'b> Lam<&'a &'b u8, App = _0>>>,
    //~^ ERROR nested quantification of lifetimes [E0316]
    _0: Debug,
{}

fn nested_bounds_desugared<_0, _1, _2, D>()
where
    D: Clone + Iterator<Item = _2>,
    _2: Send + for<'a> Iterator,
    for<'a> <_2 as Iterator>::Item: for<'b> Lam<&'a &'b u8, App = _0>,
    //~^ ERROR nested quantification of lifetimes [E0316]
    _0: Debug,
{}

fn main() {}
