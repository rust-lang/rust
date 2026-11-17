//@ check-pass

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(SingleVariantRef<'a, T>)]
enum SingleVariant<'a, T> {
    A(&'a mut T),
}

enum SingleVariantRef<'a, T> {
    A(&'a T),
}

impl<'a, T> Copy for SingleVariantRef<'a, T> {}
impl<'a, T> Clone for SingleVariantRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn single_variant(_: SingleVariantRef<'_, ()>) {}

#[derive(Reborrow, CoerceShared)]
#[coerce_shared(MultiVariantsRef<'a, 'b, T, U>)]
enum MultiVariants<'a, 'b, T, U> {
    A(&'a mut T),
    B(&'b mut U),
}

enum MultiVariantsRef<'a, 'b, T, U> {
    A(&'a T),
    B(&'b U),
}

impl<'a, 'b, T, U> Copy for MultiVariantsRef<'a, 'b, T, U> {}
impl<'a, 'b, T, U> Clone for MultiVariantsRef<'a, 'b, T, U> {
    fn clone(&self) -> Self {
        *self
    }
}

fn multi_variants(_: MultiVariantsRef<'_, '_, (), u8>, _: MultiVariantsRef<'_, '_, (), u8>) {}

#[derive(Reborrow, CoerceShared)]
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
    let a = SingleVariant::A(&mut ());
    single_variant(a);
    single_variant(a);

    let a = MultiVariants::A(&mut ());
    let b = MultiVariants::B(&mut 42);
    multi_variants(a, b);
    multi_variants(a, b);

    let a = ReorderMultiVariants::A(&mut ());
    let b = ReorderMultiVariants::B(&mut 42);
    reorder_multi_variants(a, b);
    reorder_multi_variants(a, b);
}
