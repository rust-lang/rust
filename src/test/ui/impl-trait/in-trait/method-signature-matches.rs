// edition: 2021

#![feature(return_position_impl_trait_in_trait, async_fn_in_trait)]
#![allow(incomplete_features)]

trait Uwu {
    fn owo(x: ()) -> impl Sized;
}

impl Uwu for () {
    fn owo(_: u8) {}
    //~^ ERROR method `owo` has an incompatible type for trait
}

trait AsyncUwu {
    async fn owo(x: ()) {}
}

impl AsyncUwu for () {
    async fn owo(_: u8) {}
    //~^ ERROR method `owo` has an incompatible type for trait
}

trait TooMuch {
    fn calm_down_please() -> impl Sized;
}

impl TooMuch for () {
    fn calm_down_please(_: (), _: (), _: ()) {}
    //~^ ERROR method `calm_down_please` has 3 parameters but the declaration in trait `TooMuch::calm_down_please` has 0
}

trait TooLittle {
    fn come_on_a_little_more_effort(_: (), _: (), _: ()) -> impl Sized;
}

impl TooLittle for () {
    fn come_on_a_little_more_effort() {}
    //~^ ERROR method `come_on_a_little_more_effort` has 0 parameters but the declaration in trait `TooLittle::come_on_a_little_more_effort` has 3
}

trait Lifetimes {
    fn early<'early, T>(x: &'early T) -> impl Sized;
}

impl Lifetimes for () {
    fn early<'late, T>(_: &'late ()) {}
    //~^ ERROR method `early` has an incompatible type for trait
}

fn main() {}
