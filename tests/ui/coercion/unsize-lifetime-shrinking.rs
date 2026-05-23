// Even though a type being invariant prevents shortening lifetimes via subtyping,
// unsize coercions may shorten lifetimes through an invariant type,
// but only if the appropriate CoerceUnsized impl allows it to.

use std::cell::{Cell, Ref, RefMut};

struct Thing;
trait Trait {}
impl Trait for Thing {}
trait Sub: Trait {}
impl Sub for Thing {}

//////////////////////////////////////////////////

// &T -> &U has a CoerceUnsized impl that allows shortening lifetimes

// Check that unsize-coercion works if the lifetime doesn't change.
fn ref_to_ref_same_lifetime<'a>(x: Cell<&'a Thing>) -> Cell<&'a (dyn Trait + 'static)> {
    x
}

// Sized-to-Sized unsize coercions aren't a thing, so this can't shorten the lifetime.
fn ref_to_ref_sized_to_sized<'a: 'b, 'b>(x: Cell<&'a Thing>) -> Cell<&'b Thing> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Sized-to-dyn unsize coercions can shorten the lifetime
// if the CoerceUnsized impl allows it to.
fn ref_to_ref_sized_to_dyn<'a: 'b, 'b>(x: Cell<&'a Thing>) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// Trait upcasting is an unsize coercion.
fn ref_to_ref_upcast<'a: 'b, 'b>(
    x: Cell<&'a (dyn Sub + 'static)>,
) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// An "identity" coercion from a trait object to itself is an unsize coercion.
fn ref_to_ref_dyn_same<'a: 'b, 'b>(
    x: Cell<&'a (dyn Trait + 'static)>,
) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// An unsize coercion can, surprisingly, shorten the lifetime inside a trait object.
// This ignores whether CoerceUnsized allows shortening lifetimes or not,
// since the lifetime-shortening is done "inside" Unsize, not CoerceUnsized.
fn ref_to_ref_shorten_in_dyn<'a>(x: Cell<&'a (dyn Trait + 'static)>) -> Cell<&'a (dyn Trait + 'a)> {
    x
}

//////////////////////////////////////////////////

// &mut T -> &U has a CoerceUnsized impl that allows shortening lifetimes

// Check that unsize-coercion works if the lifetime doesn't change.
fn mut_to_ref_same_lifetime<'a>(x: Cell<&'a mut Thing>) -> Cell<&'a (dyn Trait + 'static)> {
    x
}

// Sized-to-Sized unsize coercions aren't a thing, so this can't shorten the lifetime.
fn mut_to_ref_sized_to_sized<'a: 'b, 'b>(x: Cell<&'a mut Thing>) -> Cell<&'b Thing> {
    x
}
//~^^ ERROR mismatched types

// Sized-to-dyn unsize coercions can shorten the lifetime
// if the CoerceUnsized impl allows it to.
fn mut_to_ref_sized_to_dyn<'a: 'b, 'b>(x: Cell<&'a mut Thing>) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// Trait upcasting is an unsize coercion.
fn mut_to_ref_upcast<'a: 'b, 'b>(
    x: Cell<&'a mut (dyn Sub + 'static)>,
) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// An "identity" coercion from a trait object to itself is an unsize coercion.
fn mut_to_ref_dyn_same<'a: 'b, 'b>(
    x: Cell<&'a mut (dyn Trait + 'static)>,
) -> Cell<&'b (dyn Trait + 'static)> {
    x
}

// An unsize coercion can, surprisingly, shorten the lifetime inside a trait object.
// This ignores whether CoerceUnsized allows shortening lifetimes or not,
// since the lifetime-shortening is done "inside" Unsize, not CoerceUnsized.
fn mut_to_ref_shorten_in_dyn<'a>(
    x: Cell<&'a mut (dyn Trait + 'static)>,
) -> Cell<&'a (dyn Trait + 'a)> {
    x
}

//////////////////////////////////////////////////

// &mut T -> &mut U has a CoerceUnsized impl that does not allow shortening lifetimes.
// This may change in the future.

// Check that unsize-coercion works if the lifetime doesn't change.
fn mut_to_mut_same_lifetime<'a>(x: Cell<&'a mut Thing>) -> Cell<&'a mut (dyn Trait + 'static)> {
    x
}

// Sized-to-Sized unsize coercions aren't a thing, so this can't shorten the lifetime.
fn mut_to_mut_sized_to_sized<'a: 'b, 'b>(x: Cell<&'a mut Thing>) -> Cell<&'b mut Thing> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Sized-to-dyn unsize coercions can shorten the lifetime
// if the CoerceUnsized impl allows it to.
fn mut_to_mut_sized_to_dyn<'a: 'b, 'b>(
    x: Cell<&'a mut Thing>,
) -> Cell<&'b mut (dyn Trait + 'static)> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Trait upcasting is an unsize coercion.
fn mut_to_mut_upcast<'a: 'b, 'b>(
    x: Cell<&'a mut (dyn Sub + 'static)>,
) -> Cell<&'b mut (dyn Trait + 'static)> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An "identity" coercion from a trait object to itself is an unsize coercion.
fn mut_to_mut_dyn_same<'a: 'b, 'b>(
    x: Cell<&'a mut (dyn Trait + 'static)>,
) -> Cell<&'b mut (dyn Trait + 'static)> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An unsize coercion can, surprisingly, shorten the lifetime inside a trait object.
// This ignores whether CoerceUnsized allows shortening lifetimes or not,
// since the lifetime-shortening is done "inside" Unsize, not CoerceUnsized.
fn mut_to_mut_shorten_in_dyn<'a>(
    x: Cell<&'a mut (dyn Trait + 'static)>,
) -> Cell<&'a mut (dyn Trait + 'a)> {
    x
}

//////////////////////////////////////////////////

// Ref<T> -> Ref<U> has a CoerceUnsized impl that does not allow shortening lifetimes.
// This may change in the future.

// Check that unsize-coercion works if the lifetime doesn't change.
fn cell_ref_same_lifetime<'a>(x: Cell<Ref<'a, Thing>>) -> Cell<Ref<'a, dyn Trait + 'static>> {
    x
}

// Sized-to-Sized unsize coercions aren't a thing, so this can't shorten the lifetime.
fn cell_ref_sized_to_sized<'a: 'b, 'b>(x: Cell<Ref<'a, Thing>>) -> Cell<Ref<'b, Thing>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Sized-to-dyn unsize coercions can shorten the lifetime
// if the CoerceUnsized impl allows it to.
fn cell_ref_sized_to_dyn<'a: 'b, 'b>(
    x: Cell<Ref<'a, Thing>>,
) -> Cell<Ref<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Trait upcasting is an unsize coercion.
fn cell_ref_upcast<'a: 'b, 'b>(
    x: Cell<Ref<'a, dyn Sub + 'static>>,
) -> Cell<Ref<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An "identity" coercion from a trait object to itself is an unsize coercion.
fn cell_ref_dyn_same<'a: 'b, 'b>(
    x: Cell<Ref<'a, dyn Trait + 'static>>,
) -> Cell<Ref<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An unsize coercion can, surprisingly, shorten the lifetime inside a trait object.
// This ignores whether CoerceUnsized allows shortening lifetimes or not,
// since the lifetime-shortening is done "inside" Unsize, not CoerceUnsized.
fn cell_ref_shorten_in_dyn<'a>(
    x: Cell<Ref<'a, dyn Trait + 'static>>,
) -> Cell<Ref<'a, dyn Trait + 'a>> {
    x
}

//////////////////////////////////////////////////

// RefMut<T> -> RefMut<U> has a CoerceUnsized impl that does not allow shortening lifetimes.
// This may change in the future.

// Check that unsize-coercion works if the lifetime doesn't change.
fn cell_ref_mut_same_lifetime<'a>(
    x: Cell<RefMut<'a, Thing>>,
) -> Cell<RefMut<'a, dyn Trait + 'static>> {
    x
}

// Sized-to-Sized unsize coercions aren't a thing, so this can't shorten the lifetime.
fn cell_ref_mut_sized_to_sized<'a: 'b, 'b>(x: Cell<RefMut<'a, Thing>>) -> Cell<RefMut<'b, Thing>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Sized-to-dyn unsize coercions can shorten the lifetime
// if the CoerceUnsized impl allows it to.
fn cell_ref_mut_sized_to_dyn<'a: 'b, 'b>(
    x: Cell<RefMut<'a, Thing>>,
) -> Cell<RefMut<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// Trait upcasting is an unsize coercion.
fn cell_ref_mut_upcast<'a: 'b, 'b>(
    x: Cell<RefMut<'a, dyn Sub + 'static>>,
) -> Cell<RefMut<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An "identity" coercion from a trait object to itself is an unsize coercion.
fn cell_ref_mut_dyn_same<'a: 'b, 'b>(
    x: Cell<RefMut<'a, dyn Trait + 'static>>,
) -> Cell<RefMut<'b, dyn Trait + 'static>> {
    x
}
//~^^ ERROR lifetime may not live long enough

// An unsize coercion can, surprisingly, shorten the lifetime inside a trait object.
// This ignores whether CoerceUnsized allows shortening lifetimes or not,
// since the lifetime-shortening is done "inside" Unsize, not CoerceUnsized.
fn cell_ref_mut_shorten_in_dyn<'a>(
    x: Cell<RefMut<'a, dyn Trait + 'static>>,
) -> Cell<RefMut<'a, dyn Trait + 'a>> {
    x
}

//////////////////////////////////////////////////

fn main() {}
