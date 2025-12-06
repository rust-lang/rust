//@ edition:2015..2021
#![feature(iter_macro, impl_trait_in_fn_trait_return, yield_expr)]

use std::iter::iter;

fn plain() -> impl Fn() -> impl Iterator<Item = u32> {
    iter! { || {
        0.yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn arg() -> impl Fn(u32) -> impl Iterator<Item = u32> {
    iter! { |arg| {
        arg.yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn capture<'a>(a: &'a u32) -> impl Fn() -> (impl Iterator<Item = u32> + 'a) {
    iter! { || { //~ ERROR cannot return reference to function parameter `a`
        (*a).yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn capture_move(a: &u32) -> impl Fn() -> impl Iterator<Item = u32> {
    iter! { move || { //~ ERROR does not implement `Fn` because it captures
        (*a).yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn capture_move_once(a: &u32) -> impl FnOnce() -> impl Iterator<Item = u32> {
    iter! { move || {
        //~^ ERROR captures lifetime
        //~| ERROR: captures lifetime
        (*a).yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn capture_move_once_lifetimes<'a>(
    a: &'a u32,
) -> impl FnOnce() -> (impl Iterator<Item = u32> + 'a) {
    iter! { move || {
        (*a).yield;
        for x in 5..10 {
            (x * 2).yield;
        }
    } }
}

fn main() {}
