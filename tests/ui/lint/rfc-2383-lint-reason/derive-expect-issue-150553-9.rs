// Lints that are reported on a span copied from the derive input, such as a generic
// parameter or a field type, also fire in the impls derived from the item. An `#[expect]`
// on the item covers those impls too.

//@ check-pass

#[derive(Clone)]
#[expect(non_upper_case_globals)]
pub struct ConstParam<const n: usize>([u8; n]);

#[derive(Clone)]
#[expect(drop_bounds)]
pub struct DropBound<T: Drop>(T);

#[derive(Clone, Copy)]
#[expect(dyn_drop)]
pub struct DynDrop(pub *const dyn Drop);

fn main() {}
