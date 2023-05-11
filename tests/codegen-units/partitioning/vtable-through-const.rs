//

// We specify incremental here because we want to test the partitioning for
// incremental compilation
// incremental
// compile-flags:-Zprint-mono-items=lazy
// compile-flags:-Zinline-in-all-cgus

// This test case makes sure, that references made through constants are
// recorded properly in the InliningMap.

#![feature(start)]

mod mod1 {
    pub trait Trait1 {
        fn do_something(&self) {}
        fn do_something_else(&self) {}
    }

    impl Trait1 for u32 {}

    pub trait Trait1Gen<T> {
        fn do_something(&self, x: T) -> T;
        fn do_something_else(&self, x: T) -> T;
    }

    impl<T> Trait1Gen<T> for u32 {
        fn do_something(&self, x: T) -> T { x }
        fn do_something_else(&self, x: T) -> T { x }
    }

    //~ MONO_ITEM fn mod1::id::<i64> @@ vtable_through_const-mod1.volatile[Internal]
    fn id<T>(x: T) -> T { x }

    // These are referenced, so they produce mono-items (see start())
    pub const TRAIT1_REF: &'static Trait1 = &0u32 as &Trait1;
    pub const TRAIT1_GEN_REF: &'static Trait1Gen<u8> = &0u32 as &Trait1Gen<u8>;
    pub const ID_CHAR: fn(char) -> char = id::<char>;



    pub trait Trait2 {
        fn do_something(&self) {}
        fn do_something_else(&self) {}
    }

    //~ MONO_ITEM fn <u32 as mod1::Trait2>::do_something @@ vtable_through_const-mod1.volatile[Internal]
    //~ MONO_ITEM fn <u32 as mod1::Trait2>::do_something_else @@ vtable_through_const-mod1.volatile[Internal]
    impl Trait2 for u32 {}

    pub trait Trait2Gen<T> {
        fn do_something(&self, x: T) -> T;
        fn do_something_else(&self, x: T) -> T;
    }

    impl<T> Trait2Gen<T> for u32 {
        fn do_something(&self, x: T) -> T { x }
        fn do_something_else(&self, x: T) -> T { x }
    }

    // These are not referenced, so they do not produce mono-items
    pub const TRAIT2_REF: &'static Trait2 = &0u32 as &Trait2;
    pub const TRAIT2_GEN_REF: &'static Trait2Gen<u8> = &0u32 as &Trait2Gen<u8>;
    pub const ID_I64: fn(i64) -> i64 = id::<i64>;
}

//~ MONO_ITEM fn start
#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    //~ MONO_ITEM fn std::ptr::drop_in_place::<u32> - shim(None) @@ vtable_through_const[Internal]

    // Since Trait1::do_something() is instantiated via its default implementation,
    // it is considered a generic and is instantiated here only because it is
    // referenced in this module.
    //~ MONO_ITEM fn <u32 as mod1::Trait1>::do_something_else @@ vtable_through_const-mod1.volatile[External]

    // Although it is never used, Trait1::do_something_else() has to be
    // instantiated locally here too, otherwise the <&u32 as &Trait1> vtable
    // could not be fully constructed.
    //~ MONO_ITEM fn <u32 as mod1::Trait1>::do_something @@ vtable_through_const-mod1.volatile[External]
    mod1::TRAIT1_REF.do_something();

    // Same as above
    //~ MONO_ITEM fn <u32 as mod1::Trait1Gen<u8>>::do_something @@ vtable_through_const-mod1.volatile[External]
    //~ MONO_ITEM fn <u32 as mod1::Trait1Gen<u8>>::do_something_else @@ vtable_through_const-mod1.volatile[External]
    //~ MONO_ITEM fn <u32 as mod1::Trait2Gen<u8>>::do_something @@ vtable_through_const-mod1.volatile[Internal]
    //~ MONO_ITEM fn <u32 as mod1::Trait2Gen<u8>>::do_something_else @@ vtable_through_const-mod1.volatile[Internal]
    mod1::TRAIT1_GEN_REF.do_something(0u8);

    //~ MONO_ITEM fn mod1::id::<char> @@ vtable_through_const-mod1.volatile[External]
    mod1::ID_CHAR('x');

    0
}
