#![feature(intrinsics, rustc_attrs)]

// Intrinsics are the only (?) extern blocks supporting generics.
// Once intrinsics have to be declared via `#[rustc_intrinsic]`,
// the entire support for generics in extern fn can probably be removed.

extern "rust-intrinsic" {
    // Silent bounds made explicit to make sure they are actually
    // resolved.
    fn transmute<T: Sized, U: Sized>(val: T) -> U;

    // Bounds aren't checked right now, so this should work
    // even though it's incorrect.
    fn size_of_val<T: Clone>(x: *const T) -> usize;

    // Unresolved bounds should still error.
    fn align_of<T: NoSuchTrait>() -> usize;
    //~^ ERROR cannot find trait `NoSuchTrait` in this scope
}

fn main() {}
