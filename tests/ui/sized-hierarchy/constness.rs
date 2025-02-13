//@ compile-flags: -Zunstable-options
//@ edition: future
#![allow(internal_features)]
#![feature(rustc_attrs, const_trait_impl, sized_hierarchy)]

use std::marker::MetaSized;

#[rustc_non_const_sized]
struct NotConstSized;

#[rustc_non_const_sized]
struct NotConstMetaSized([u8]);

impl NotConstMetaSized {
    fn new() -> &'static Self { unimplemented!() }
}

const fn size_of<T: ~const Sized>() { unimplemented!() }
const fn size_of_val<T: ~const MetaSized>(_t: &T) { unimplemented!() }

fn main() {
    let _ = const { size_of::<NotConstSized>() };
//~^ ERROR the trait bound `NotConstSized: const Sized` is not satisfied
    let _ = size_of::<NotConstSized>();

    let _ = const { size_of::<NotConstMetaSized>() };
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    let _ = size_of::<NotConstMetaSized>();
//~^ ERROR the size for values of type `[u8]` cannot be known at compilation time

    let v = NotConstSized;
    let _ = const { size_of_val(&v) };
//~^ ERROR attempt to use a non-constant value in a constant
    let _ = size_of_val(&v);

    let v = NotConstMetaSized::new();
    let _ = const { size_of_val(v) };
//~^ ERROR attempt to use a non-constant value in a constant
    let _ = size_of_val(v);
}
