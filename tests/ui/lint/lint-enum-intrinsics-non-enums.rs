// Test the enum_intrinsics_non_enums lint.

#![feature(variant_count)]

use std::mem::{discriminant, variant_count};

enum SomeEnum {
    A,
    B,
}

struct SomeStruct;

fn generic_discriminant<T>(v: &T) {
    discriminant::<T>(v);
}

fn generic_variant_count<T>() -> usize {
    variant_count::<T>()
}

fn test_discriminant() {
    discriminant(&SomeEnum::A);
    generic_discriminant(&SomeEnum::B);

    discriminant(&());
    //~^ error: the return value of `mem::discriminant` is unspecified when called with a non-enum type

    discriminant(&&SomeEnum::B);
    //~^ error: the return value of `mem::discriminant` is unspecified when called with a non-enum type

    discriminant(&SomeStruct);
    //~^ error: the return value of `mem::discriminant` is unspecified when called with a non-enum type

    discriminant(&123u32);
    //~^ error: the return value of `mem::discriminant` is unspecified when called with a non-enum type

    discriminant(&&123i8);
    //~^ error: the return value of `mem::discriminant` is unspecified when called with a non-enum type
}

fn test_variant_count() {
    variant_count::<SomeEnum>();
    generic_variant_count::<SomeEnum>();

    variant_count::<&str>();
    //~^ error: the return value of `mem::variant_count` is unspecified when called with a non-enum type

    variant_count::<*const u8>();
    //~^ error: the return value of `mem::variant_count` is unspecified when called with a non-enum type

    variant_count::<()>();
    //~^ error: the return value of `mem::variant_count` is unspecified when called with a non-enum type

    variant_count::<&SomeEnum>();
    //~^ error: the return value of `mem::variant_count` is unspecified when called with a non-enum type
}

fn main() {
    test_discriminant();
    test_variant_count();

    // The lint ignores cases where the type is generic, so these should be
    // allowed even though their return values are unspecified
    generic_variant_count::<SomeStruct>();
    generic_discriminant::<SomeStruct>(&SomeStruct);
}
