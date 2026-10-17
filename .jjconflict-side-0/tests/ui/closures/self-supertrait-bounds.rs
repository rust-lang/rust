//@ check-pass

// Makes sure that we only consider `Self` supertrait predicates while
// elaborating during closure signature deduction.

#![feature(trait_alias)]

trait Confusing<F> = Fn(i32) where F: Fn(u32);

fn alias<T: Confusing<F>, F>(_: T, _: F) {}

fn main() {
    alias(|_| {}, |_| {});
}
