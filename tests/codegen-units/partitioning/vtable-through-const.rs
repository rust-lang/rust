//@ incremental
// Need to disable optimizations to ensure consistent output across all CI runners.
//@ compile-flags: -Copt-level=0

#![crate_type = "rlib"]

// This test case makes sure that references made through constants cause trait associated methods
// to be monomorphized when required.

mod mod1 {
    struct NeedsDrop;

    impl Drop for NeedsDrop {
        fn drop(&mut self) {}
    }

    pub trait Trait1 {
        fn do_something(&self) {}
        fn do_something_else(&self) {}
    }

    impl Trait1 for NeedsDrop {}

    pub trait Trait1Gen<T> {
        fn do_something(&self, x: T) -> T;
        fn do_something_else(&self, x: T) -> T;
    }

    impl<T> Trait1Gen<T> for NeedsDrop {
        fn do_something(&self, x: T) -> T {
            x
        }
        fn do_something_else(&self, x: T) -> T {
            x
        }
    }

    fn id<T>(x: T) -> T {
        x
    }

    // These are referenced, so they produce mono-items (see main)
    pub const TRAIT1_REF: &'static Trait1 = &NeedsDrop as &Trait1;
    pub const TRAIT1_GEN_REF: &'static Trait1Gen<u8> = &NeedsDrop as &Trait1Gen<u8>;
    pub const ID_CHAR: fn(char) -> char = id::<char>;

    pub trait Trait2 {
        fn do_something(&self) {}
        fn do_something_else(&self) {}
    }

    impl Trait2 for NeedsDrop {}

    pub trait Trait2Gen<T> {
        fn do_something(&self, x: T) -> T;
        fn do_something_else(&self, x: T) -> T;
    }

    impl<T> Trait2Gen<T> for NeedsDrop {
        fn do_something(&self, x: T) -> T {
            x
        }
        fn do_something_else(&self, x: T) -> T {
            x
        }
    }

    // These are not referenced, so they do not produce mono-items
    pub const TRAIT2_REF: &'static Trait2 = &NeedsDrop as &Trait2;
    pub const TRAIT2_GEN_REF: &'static Trait2Gen<u8> = &NeedsDrop as &Trait2Gen<u8>;
    pub const ID_I64: fn(i64) -> i64 = id::<i64>;
}

//~ MONO_ITEM fn main @@ vtable_through_const[External]
pub fn main() {
    //~ MONO_ITEM fn <mod1::NeedsDrop as std::ops::Drop>::drop @@ vtable_through_const-fallback.cgu[External]
    //~ MONO_ITEM fn std::ptr::drop_in_place::<mod1::NeedsDrop> - shim(Some(mod1::NeedsDrop)) @@ vtable_through_const-fallback.cgu[External]

    // Since Trait1::do_something() is instantiated via its default implementation,
    // it is considered a generic and is instantiated here only because it is
    // referenced in this module.
    //~ MONO_ITEM fn <mod1::NeedsDrop as mod1::Trait1>::do_something_else @@ vtable_through_const-mod1.volatile[External]

    // Although it is never used, Trait1::do_something_else() has to be
    // instantiated locally here too, otherwise the <&NeedsDrop as &Trait1> vtable
    // could not be fully constructed.
    //~ MONO_ITEM fn <mod1::NeedsDrop as mod1::Trait1>::do_something @@ vtable_through_const-mod1.volatile[External]
    mod1::TRAIT1_REF.do_something();

    // Same as above
    //~ MONO_ITEM fn <mod1::NeedsDrop as mod1::Trait1Gen<u8>>::do_something @@ vtable_through_const-mod1.volatile[External]
    //~ MONO_ITEM fn <mod1::NeedsDrop as mod1::Trait1Gen<u8>>::do_something_else @@ vtable_through_const-mod1.volatile[External]
    mod1::TRAIT1_GEN_REF.do_something(0u8);

    //~ MONO_ITEM fn mod1::id::<char> @@ vtable_through_const-mod1.volatile[External]
    mod1::ID_CHAR('x');
}
