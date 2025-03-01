//@ check-pass

#![feature(trait_alias)]

trait Confusing<F> = Fn(i32) where F: Fn(u32);

fn alias<T: Confusing<F>, F>(_: T, _: F) {}

fn main() {
    alias(|_| {}, |_| {});
}
