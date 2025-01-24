// Projections cover type parameters if they normalize to a (local) type that covers them.
// This ensures that we don't perform an overly strict check on
// projections like in closed PR #100555 which did a syntactic
// check for type parameters in projections without normalizing
// first which would've lead to real-word regressions.

//@ check-pass
//@ aux-crate:foreign=parametrized-trait.rs
//@ edition:2021

trait Project { type Output; }

impl<T> Project for Wrapper<T> {
    type Output = Local;
}

struct Wrapper<T>(T);
struct Local;

impl<T> foreign::Trait1<Local, T> for <Wrapper<T> as Project>::Output {}

fn main() {}
