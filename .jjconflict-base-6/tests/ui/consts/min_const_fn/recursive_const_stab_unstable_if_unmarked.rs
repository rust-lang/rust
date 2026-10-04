//@ compile-flags: -Zforce-unstable-if-unmarked
//@ edition: 2021
#![feature(const_async_blocks, rustc_attrs)]

pub const fn not_stably_const() {
    // We need to do something const-unstable here.
    // For now we use `async`, eventually we might have to add a auxiliary crate
    // as a dependency just to be sure we have something const-unstable.
    let _x = async { 15 };
}

#[rustc_const_stable_indirect]
pub const fn expose_on_stable() {
    // Calling `not_stably_const` here is *not* okay.
    not_stably_const();
    //~^ERROR: cannot use `#[feature(rustc_private)]`
    // Also directly using const-unstable things is not okay.
    let _x = async { 15 };
    //~^ERROR: cannot use `#[feature(const_async_blocks)]`
}

fn main() {
    const { expose_on_stable() };
}
