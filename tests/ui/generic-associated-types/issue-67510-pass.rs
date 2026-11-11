//@ check-fail

trait X {
    type Y<'a>;
}

fn _func1<'a>(_x: Box<dyn X<Y<'a>=&'a ()>>) {}
//~^ ERROR the trait `X` is not dyn compatible

fn main() {}
