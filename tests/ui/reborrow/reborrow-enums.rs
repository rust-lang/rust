//@ check-pass

#![feature(reborrow)]

use std::marker::Reborrow;

#[derive(Reborrow)]
enum SingleVariant<'a, T> {
    A(&'a mut T),
}

fn single_variant(_: SingleVariant<'_, ()>) {}

#[derive(Reborrow)]
enum MultiVariants<'a, 'b, T, U> {
    A(&'a mut T),
    B(&'b mut U),
}

fn multi_variants(_: MultiVariants<'_, '_, (), u8>, _: MultiVariants<'_, '_, (), u8>) {}

#[derive(Reborrow)]
enum NestedEnum<'a, 'b, T, U> {
    Single(SingleVariant<'a, T>),
    Multi(MultiVariants<'a, 'b, T, U>),
}

fn nested_enum(
    _: NestedEnum<'_, '_, (), u8>,
    _: NestedEnum<'_, '_, (), u8>,
    _: NestedEnum<'_, '_, (), u8>,
) {}

#[derive(Reborrow)]
enum ReorderEnum<'a, 'b, T, U> {
    Multi(MultiVariants<'a, 'b, T, U>),
    Single(SingleVariant<'a, T>),
}

fn nested_enum_reorder(
    _: ReorderEnum<'_, '_, (), u8>,
    _: ReorderEnum<'_, '_, (), u8>,
    _: ReorderEnum<'_, '_, (), u8>,
) {}

fn main() {
    let a = SingleVariant::A(&mut ());
    single_variant(a);
    single_variant(a);

    let a = MultiVariants::A(&mut ());
    let b = MultiVariants::B(&mut 42);
    multi_variants(a, b);
    multi_variants(a, b);

    let sa = NestedEnum::Single(SingleVariant::A(&mut ()));
    let ma = NestedEnum::Multi(MultiVariants::A(&mut ()));
    let mb = NestedEnum::Multi(MultiVariants::B(&mut 42));
    nested_enum(sa, ma, mb);
    nested_enum(sa, ma, mb);

    let sa = ReorderEnum::Single(SingleVariant::A(&mut ()));
    let ma = ReorderEnum::Multi(MultiVariants::A(&mut ()));
    let mb = ReorderEnum::Multi(MultiVariants::B(&mut 42));
    nested_enum_reorder(sa, ma, mb);
    nested_enum_reorder(sa, ma, mb);
}
