#![feature(iter_macro, impl_trait_in_fn_trait_return, yield_expr)]

use std::iter::iter;

fn plain() -> impl Fn() -> impl Iterator<Item = u32> {
    iter! { || {
        yield 0;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn arg() -> impl Fn(u32) -> impl Iterator<Item = u32> {
    iter! { |arg| {
        yield arg;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn capture<'a>(a: &'a u32) -> impl Fn() -> (impl Iterator<Item = u32> + 'a) {
    iter! { || { //~ ERROR cannot return reference to function parameter `a`
        yield *a;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn capture_move(a: &u32) -> impl Fn() -> impl Iterator<Item = u32> {
    iter! { move || { //~ ERROR does not implement `Fn` because it captures
        yield *a;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn capture_move_once(a: &u32) -> impl FnOnce() -> impl Iterator<Item = u32> {
    iter! { move || {
        //~^ ERROR captures lifetime
        //~| ERROR: captures lifetime
        yield *a;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn capture_move_once_lifetimes<'a>(
    a: &'a u32,
) -> impl FnOnce() -> (impl Iterator<Item = u32> + 'a) {
    iter! { move || {
        yield *a;
        for x in 5..10 {
            yield x * 2;
        }
    } }
}

fn main() {}
