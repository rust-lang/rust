#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[derive(Reborrow, CoerceShared)] //~ ERROR implementing `CoerceShared` requires source and target types to use the same field style
#[coerce_shared(ReorderMultiVariantsRef<'a, 'b, T, U>)]
enum ReorderMultiVariants<'a, 'b, T, U> {
    A(&'a mut T),
    B(&'b mut U),
}

enum ReorderMultiVariantsRef<'a, 'b, T, U> {
    B(&'b U),
    A(&'a T),
}

impl<'a, 'b, T, U> Copy for ReorderMultiVariantsRef<'a, 'b, T, U> {}
impl<'a, 'b, T, U> Clone for ReorderMultiVariantsRef<'a, 'b, T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

fn reorder_multi_variants(
    _: ReorderMultiVariantsRef<'_, '_, (), u8>,
    _: ReorderMultiVariantsRef<'_, '_, (), u8>,
) {}

fn main() {
    let a = ReorderMultiVariants::A(&mut ());
    let b = ReorderMultiVariants::B(&mut 42);
    reorder_multi_variants(a, b);
    reorder_multi_variants(a, b);
}
