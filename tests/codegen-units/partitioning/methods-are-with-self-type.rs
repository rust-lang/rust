// We specify incremental here because we want to test the partitioning for incremental compilation
//@ incremental
//@ compile-flags:-Zprint-mono-items=lazy

#![crate_type = "lib"]

pub struct SomeType;

struct SomeGenericType<T1, T2>(T1, T2);

pub mod mod1 {
    use super::{SomeGenericType, SomeType};

    // Even though the impl is in `mod1`, the methods should end up in the
    // parent module, since that is where their self-type is.
    impl SomeType {
        //~ MONO_ITEM fn mod1::<impl SomeType>::method @@ methods_are_with_self_type[External]
        pub fn method(&self) {}
        //~ MONO_ITEM fn mod1::<impl SomeType>::associated_fn @@ methods_are_with_self_type[External]
        pub fn associated_fn() {}
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
impl<T> Trait for T {
    fn foo(&self) {}
}

mod type1 {
    pub struct Struct;
}

mod type2 {
    pub struct Struct;
}

//~ MONO_ITEM fn start @@ methods_are_with_self_type[External]
pub fn start() {
    //~ MONO_ITEM fn mod1::<impl SomeGenericType<u32, u64>>::method @@ methods_are_with_self_type.volatile[External]
    SomeGenericType(0u32, 0u64).method();
    //~ MONO_ITEM fn mod1::<impl SomeGenericType<char, &str>>::associated_fn @@ methods_are_with_self_type.volatile[External]
    SomeGenericType::associated_fn('c', "&str");

    //~ MONO_ITEM fn <type1::Struct as Trait>::foo @@ methods_are_with_self_type-type1.volatile[External]
    type1::Struct.foo();
    //~ MONO_ITEM fn <type2::Struct as Trait>::foo @@ methods_are_with_self_type-type2.volatile[External]
    type2::Struct.foo();

    //~ MONO_ITEM fn <type1::Struct as Trait>::default @@ methods_are_with_self_type-type1.volatile[External]
    type1::Struct.default();
    //~ MONO_ITEM fn <type2::Struct as Trait>::default @@ methods_are_with_self_type-type2.volatile[External]
    type2::Struct.default();
}
