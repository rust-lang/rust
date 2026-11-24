//! A `!Move` type from a dependency is not `Move` without the feature.
//@ aux-build: move-trait-upstream.rs
//@ compile-flags: -Znext-solver

extern crate move_trait_upstream as up;

struct Local;

fn generic<T>(t: T) {
    up::need((t,));
}

fn main() {
    up::need(Local);
    generic(Local);
    up::need(up::NotMove);
    //~^ ERROR values of type `NotMove` may not be movable
    up::need((up::NotMove,));
    //~^ ERROR values of type `NotMove` may not be movable
    up::need([up::NotMove]);
    //~^ ERROR values of type `NotMove` may not be movable
}
