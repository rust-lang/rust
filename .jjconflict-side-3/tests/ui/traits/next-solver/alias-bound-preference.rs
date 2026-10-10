//@ revisions: old next
//@[next] compile-flags: -Znext-solver
//@ run-pass

// A test for https://github.com/rust-lang/trait-system-refactor-initiative/issues/45.

trait Trait {
    type Assoc: Into<u32>;
}
impl<T: Into<u32>> Trait for T {
    type Assoc = T;
}
fn prefer_alias_bound_projection<T: Trait>(x: T::Assoc) {
    // There are two possible types for `x`:
    // - `u32` by using the "alias bound" of `<T as Trait>::Assoc`
    // - `<T as Trait>::Assoc`, i.e. `u16`, by using `impl<T> From<T> for T`
    //
    // We infer the type of `x` to be `u32` here as it is highly likely
    // that this is expected by the user.
    let x = x.into();
    assert_eq!(std::mem::size_of_val(&x), 4);
}

fn impl_trait() -> impl Into<u32> {
    0u16
}

fn main() {
    // There are two possible types for `x`:
    // - `u32` by using the "alias bound" of `impl Into<u32>`
    // - `impl Into<u32>`, i.e. `u16`, by using `impl<T> From<T> for T`
    //
    // We infer the type of `x` to be `u32` here as it is highly likely
    // that this is expected by the user.
    let x = impl_trait().into();
    assert_eq!(std::mem::size_of_val(&x), 4);

    prefer_alias_bound_projection::<u16>(1);
}
