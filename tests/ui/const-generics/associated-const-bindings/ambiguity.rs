// We used to say "ambiguous associated type" on ambiguous associated consts.
// Ensure that we now use the correct label.

#![feature(adt_const_params, min_generic_const_args, unsized_const_params)]
#![allow(incomplete_features)]

trait Trait0: Parent0<i32> + Parent0<u32> {}
trait Parent0<T> {
    #[rustc_always_gca]
    const K: ();
}

fn take0(_: impl Trait0<K = const {}>) {}
//~^ ERROR ambiguous associated constant `K` in bounds of `Trait0`

trait Trait1: Parent1 + Parent2 {}
trait Parent1 {
    #[rustc_always_gca]
    const C: i32;
}
trait Parent2 {
    #[rustc_always_gca]
    const C: &'static str;
}

fn take1(_: impl Trait1<C = { core::direct_const_arg!("?") }>) {}
//~^ ERROR ambiguous associated constant `C` in bounds of `Trait1`

fn main() {}
