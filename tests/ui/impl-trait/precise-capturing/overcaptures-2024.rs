//@ run-rustfix
//@ edition: 2018

#![allow(unused)]
#![deny(impl_trait_overcaptures)]

fn named<'a>(x: &'a i32) -> impl Sized { *x }
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

fn implicit(x: &i32) -> impl Sized { *x }
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

struct W;
impl W {
    fn hello(&self, x: &i32) -> impl Sized + '_ { self }
    //~^ ERROR `impl Sized + '_` will capture more lifetimes than possibly intended in edition 2024
    //~| WARN this changes meaning in Rust 2024
}

trait Higher<'a> {
    type Output;
}
impl Higher<'_> for () {
    type Output = ();
}

fn hrtb() -> impl for<'a> Higher<'a, Output = impl Sized> {}
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

fn apit(_: &impl Sized) -> impl Sized {}
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

fn apit2<U>(_: &impl Sized, _: U) -> impl Sized {}
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

async fn async_fn<'a>(x: &'a ()) -> impl Sized {}
//~^ ERROR `impl Sized` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

pub fn parens(x: &i32) -> &impl Clone { x }
//~^ ERROR `impl Clone` will capture more lifetimes than possibly intended in edition 2024
//~| WARN this changes meaning in Rust 2024

fn main() {}
