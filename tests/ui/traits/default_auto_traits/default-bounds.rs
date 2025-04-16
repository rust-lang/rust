//@ compile-flags: -Zexperimental-default-bounds

#![feature(
    auto_traits,
    lang_items,
    negative_impls,
    no_core,
    rustc_attrs
)]
#![allow(incomplete_features)]
#![no_std]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
pub trait Copy {}

#[lang = "default_trait1"]
auto trait Leak {}

#[lang = "default_trait2"]
auto trait SyncDrop {}

struct Forbidden;

impl !Leak for Forbidden {}
impl !SyncDrop for Forbidden {}

struct Accepted;

fn bar<T: Leak>(_: T) {}

fn main() {
    // checking that bounds can be added explicitly
    bar(Forbidden);
    //~^ ERROR the trait bound `Forbidden: Leak` is not satisfied
    //~| ERROR the trait bound `Forbidden: SyncDrop` is not satisfied
    bar(Accepted);
}
