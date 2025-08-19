//@ revisions: stable deref_patterns
//@[deref_patterns] check-pass
//! `deref_patterns` allows string and byte string literal patterns to implicitly peel references
//! and smart pointers from the scrutinee before matching. Since strings and byte strings themselves
//! have reference types, we need to make sure we don't peel too much. By leaving the type of the
//! match scrutinee partially uninferred, these tests make sure we only peel as much as needed in
//! order to match. In particular, when peeling isn't needed, the results should be the same was
//! we'd get without `deref_patterns` enabled.

#![cfg_attr(deref_patterns, feature(deref_patterns))]
#![cfg_attr(deref_patterns, expect(incomplete_features))]

fn uninferred<T>() -> T { unimplemented!() }

// Assert type equality without allowing coercions.
trait Is<T> {}
impl<T> Is<T> for T {}
fn has_type<T>(_: impl Is<T>) {}

fn main() {
    // We don't need to peel anything to unify the type of `x` with `&str`, so `x: &str`.
    let x = uninferred();
    if let "..." = x {}
    has_type::<&str>(x);

    // We don't need to peel anything to unify the type of `&x` with `&[u8; 3]`, so `x: [u8; 3]`.
    let x = uninferred();
    if let b"..." = &x {}
    has_type::<[u8; 3]>(x);

    // Peeling a single `&` lets us unify the type of `&x` with `&[u8; 3]`, giving `x: [u8; 3]`.
    let x = uninferred();
    if let b"..." = &&x {}
    //[stable]~^ ERROR: mismatched types
    has_type::<[u8; 3]>(x);

    // We have to peel both the `&` and the box before unifying the type of `x` with `&str`.
    let x = uninferred();
    if let "..." = &Box::new(x) {}
    //[stable]~^ ERROR mismatched types
    has_type::<&str>(x);

    // After peeling the box, we can unify the type of `&x` with `&[u8; 3]`, giving `x: [u8; 3]`.
    let x = uninferred();
    if let b"..." = Box::new(&x) {}
    //[stable]~^ ERROR mismatched types
    has_type::<[u8; 3]>(x);

    // `&` and `&mut` aren't interchangeable: `&mut`s need to be peeled before unifying, like boxes:
    let mut x = uninferred();
    if let "..." = &mut x {}
    //[stable]~^ ERROR mismatched types
    has_type::<&str>(x);
}
