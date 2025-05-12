//@aux-build:dummy.rs

#![forbid(unstable_features)]

#![allow()] // OK

#[allow()] // OK
extern crate dummy;

extern crate core;

#[allow()] // OK
pub mod empty_crate;

#[allow()] // OK
pub mod module {
    #![allow()] // OK

    #[allow()] // OK
    pub static GLOBAL: u32 = 42;

    #[allow()] // OK
    pub static mut GLOBAL_MUT: u32 = 42;

    #[allow()] // OK
    pub const CONST: u32 = 42;

    #[allow()] // OK
    use core::fmt::Debug;

    #[allow()] // OK
    trait TraitAlias = Debug; //~ ERROR trait aliases are experimental

    #[allow()] // OK
    pub type TypeAlias = u32;

    #[allow()] // OK
    pub struct Struct<
        #[allow()] 'lt, // OK
        #[allow()] T> // OK
     {
        #[allow()] // OK
        pub struct_field: u32,
        #[allow()] // OK
        pub generic_field: &'lt T,
    }

    #[allow()] // OK
    pub struct TupleStruct(#[allow()] pub u32); // OK

    #[allow()] // OK
    pub enum Enum {
        #[allow()] // OK
        FieldVariant {
            #[allow()] // OK
            field: u32,
        },
        #[allow()] // OK
        TupleVariant(#[allow()] u32),
    }
    #[allow()] // OK
    pub union Union {
        #[allow()] // OK
        pub field: u32,
    }

    #[allow()] // OK
    impl<'lt, T> Struct<'lt, T> {
        #![allow()] // OK
        #[allow()] // OK
        pub fn method(#[allow()] &mut self) {  // OK

            #[allow()] //~ ERROR
            self.struct_field = 73;
        }
        #[allow()] // OK
        pub fn static_method(#[allow()] _arg: u32) {}
    }

    #[allow()]
    impl<
        #[allow()] 'lt, // OK
        #[allow()] T // OK
    > Drop for Struct<'lt, T> {
        #[allow()] // OK
        fn drop(&mut self) {
            // ..
        }
    }

    #[allow()]  // OK
    pub fn function<
        #[allow()] const N: usize // OK
    >(#[allow()] arg: u32) -> u32 { // OK
        #![allow()] // OK
        arg
    }

    #[allow()] // OK
    pub trait Trait {
        #[allow()] // OK
        fn trait_method() {}
    }
}

#[allow()] // OK
fn main() {
    let _closure = #[allow()] //~ ERROR attributes on expressions are experimental
    |#[allow()] _arg: u32| {}; // OK

    let _move_closure = #[allow()] //~ ERROR attributes on expressions are experimental
    move |#[allow()] _arg: u32| {}; // OK

    #[allow()] // OK
    {
        #![allow()] // OK

        #[allow()] // OK
        let variable = 42_u32;

        let _array = #[allow()] //~ ERROR attributes on expressions are experimental
        [
            #[allow()] // OK
            1,
            2,
        ];
        let _tuple = #[allow()] //~ ERROR attributes on expressions are experimental
        (
            #[allow()] // OK
            1,
            2,
        );
        let _tuple_struct = module::TupleStruct(
            #[allow()] // OK
            2,
        );
        let _struct = module::Struct {
            #[allow()] // OK
            struct_field: 42,
            generic_field: &13,
        };

        let _union = module::Union {
            #[allow()] // OK
            field: 42,
        };

        let _fn_call = module::function::<7>(
            #[allow()] // OK
            42,
        );

        #[allow()] // OK
        match variable {
            #[allow()] // OK
            _match_arm => {}
        }

        let tail = ();

        #[allow()] // OK
        tail
    }

    {
        fn f(_: impl Fn()) {}
        // Is this attribute argument-level or expression-level?
        f(
            #[allow()] // OK
            || {},
        );
    }

    #[allow()] // OK
    unsafe {
        let _x = [1, 2, 3].get_unchecked(1);
    }
}

#[allow()] // OK
unsafe extern "C" {
    #![allow()] // OK

    #[allow()] // OK
    pub fn external_function(
        #[allow()] arg: *mut u8, // OK
        #[allow()] ... // OK
    );

    #[allow()] // OK
    pub static EXTERNAL_STATIC: *const u32;

    #[allow()] // OK
    pub static mut EXTERNAL_STATIC_MUT: *mut u32;
}

#[allow()] // OK
pub unsafe extern "C" fn abi_function(#[allow()] _: u32) {}

#[allow()] // OK
#[macro_export]
macro_rules! my_macro {
    () => {};
}

#[allow()] // OK
#[test]
fn test_fn() {}
