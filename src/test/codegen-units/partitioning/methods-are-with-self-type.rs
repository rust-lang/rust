// Currently, all generic functions are instantiated in each codegen unit that
// uses them, even those not marked with #[inline], so this test does not make
// much sense at the moment.
// ignore-test

// ignore-tidy-linelength
// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-mono-items=lazy -Zincremental=tmp/partitioning-tests/methods-are-with-self-type

#![allow(dead_code)]
#![feature(start)]

struct SomeType;

struct SomeGenericType<T1, T2>(T1, T2);

mod mod1 {
    use super::{SomeType, SomeGenericType};

    // Even though the impl is in `mod1`, the methods should end up in the
    // parent module, since that is where their self-type is.
    impl SomeType {
        //~ MONO_ITEM fn methods_are_with_self_type::mod1[0]::{{impl}}[0]::method[0] @@ methods_are_with_self_type[External]
        fn method(&self) {}

        //~ MONO_ITEM fn methods_are_with_self_type::mod1[0]::{{impl}}[0]::associated_fn[0] @@ methods_are_with_self_type[External]
        fn associated_fn() {}
    }

    impl<T1, T2> SomeGenericType<T1, T2> {
        pub fn method(&self) {}
        pub fn associated_fn(_: T1, _: T2) {}
    }
}

trait Trait {
    fn foo(&self);
    fn default(&self) {}
}

// We provide an implementation of `Trait` for all types. The corresponding
// monomorphizations should end up in whichever module the concrete `T` is.
impl<T> Trait for T
{
    fn foo(&self) {}
}

mod type1 {
    pub struct Struct;
}

mod type2 {
    pub struct Struct;
}

//~ MONO_ITEM fn methods_are_with_self_type::start[0]
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn methods_are_with_self_type::mod1[0]::{{impl}}[1]::method[0]<u32, u64> @@ methods_are_with_self_type.volatile[WeakODR]
    SomeGenericType(0u32, 0u64).method();
    //~ MONO_ITEM fn methods_are_with_self_type::mod1[0]::{{impl}}[1]::associated_fn[0]<char, &str> @@ methods_are_with_self_type.volatile[WeakODR]
    SomeGenericType::associated_fn('c', "&str");

    //~ MONO_ITEM fn methods_are_with_self_type::{{impl}}[0]::foo[0]<methods_are_with_self_type::type1[0]::Struct[0]> @@ methods_are_with_self_type-type1.volatile[WeakODR]
    type1::Struct.foo();
    //~ MONO_ITEM fn methods_are_with_self_type::{{impl}}[0]::foo[0]<methods_are_with_self_type::type2[0]::Struct[0]> @@ methods_are_with_self_type-type2.volatile[WeakODR]
    type2::Struct.foo();

    //~ MONO_ITEM fn methods_are_with_self_type::Trait[0]::default[0]<methods_are_with_self_type::type1[0]::Struct[0]> @@ methods_are_with_self_type-type1.volatile[WeakODR]
    type1::Struct.default();
    //~ MONO_ITEM fn methods_are_with_self_type::Trait[0]::default[0]<methods_are_with_self_type::type2[0]::Struct[0]> @@ methods_are_with_self_type-type2.volatile[WeakODR]
    type2::Struct.default();

    0
}

//~ MONO_ITEM drop-glue i8
