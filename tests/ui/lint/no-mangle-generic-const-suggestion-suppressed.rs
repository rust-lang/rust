//! Ensure the `no_mangle_const_items` lint triggers but does not offer a `pub static`
//! suggestion for consts that have generics or a where-clause.
//! regression test for <https://github.com/rust-lang/rust/issues/149511>

#![feature(generic_const_items)]
#![allow(incomplete_features)]
#![deny(no_mangle_const_items)]
trait Trait {
    const ASSOC: u32;
}

#[unsafe(no_mangle)]
const WHERE_BOUND: u32 = <&'static ()>::ASSOC where for<'a> &'a (): Trait;
//~^ ERROR: const items should never be `#[no_mangle]`

#[no_mangle]
const _: () = () where;
//~^ ERROR: const items should never be `#[no_mangle]`

#[unsafe(no_mangle)]
pub const GENERIC<const N: usize>: usize = N;
//~^ ERROR: const items should never be `#[no_mangle]`

fn main() {}
